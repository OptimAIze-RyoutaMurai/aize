"""WebSocket peer connection handler for AIze-to-AIze sessions.

Protocol (JSON over WebSocket text frames):

  Client → Server
  ---------------
  {"type": "auth", "username": "...", "password": "..."}
  {"type": "list_open_sessions"}
  {"type": "join_session",  "username": "...", "session_id": "..."}
  {"type": "leave_session", "username": "...", "session_id": "..."}
  {"type": "message",       "username": "...", "session_id": "...", "text": "..."}
  {"type": "ping"}

  Server → Client
  ---------------
  {"type": "auth_ok",        "node_id": "...", "peer_id": "...", "run_id": "..."}
  {"type": "auth_error",     "message": "..."}
  {"type": "open_sessions",  "sessions": [...]}
  {"type": "session_joined", "username": "...", "session_id": "..."}
  {"type": "session_left",   "username": "...", "session_id": "..."}
  {"type": "session_event",  "username": "...", "session_id": "...", "entry": {...}}
  {"type": "message_accepted","username": "...", "session_id": "..."}
  {"type": "pong"}
  {"type": "error",          "message": "..."}
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, Callable

from runtime.ws_bridge import (
    OP_CLOSE,
    OP_PING,
    OP_TEXT,
    read_frame,
    write_close_frame,
    write_pong_frame,
    write_text_frame,
)
from wire.protocol import utc_ts


def handle_peer_connection(
    *,
    rfile: Any,
    wfile: Any,
    runtime_root: Path,
    manifest: dict[str, Any],
    self_service: dict[str, Any],
    process_id: str,
    log_path: Path,
    default_target: str,
    default_provider: str,
    codex_service_pool: list[str],
    claude_service_pool: list[str],
    # Shared state / callbacks (matching cli_service_adapter closures)
    append_history: Callable,
    send_router_control: Callable,
    make_dispatch_pending_message: Callable,
    make_aize_pending_input: Callable,
    append_pending_input: Callable,
    verify_user_password: Callable,
    issue_auth_context: Callable,
    get_session_service: Callable,
    lease_session_service: Callable,
    list_peer_joinable_sessions: Callable,
    register_history_subscriber: Callable,
    unregister_history_subscriber: Callable,
    write_jsonl: Callable,
) -> None:
    """Run the WebSocket peer session loop until the connection closes."""
    _lock = threading.Lock()
    _closed = threading.Event()
    _auth_context: dict[str, Any] | None = None
    # key → subscriber queue  (key = "username::session_id")
    _subscriptions: dict[str, queue.Queue[dict[str, Any]]] = {}

    # ------------------------------------------------------------------ helpers

    def _send(msg: dict[str, Any]) -> bool:
        try:
            write_text_frame(wfile, json.dumps(msg, ensure_ascii=False))
            return True
        except Exception:
            _closed.set()
            return False

    def _event_pump(key: str, username: str, session_id: str, sub_q: "queue.Queue[dict]") -> None:
        """Forward history entries to the remote peer (runs in its own thread)."""
        while not _closed.is_set():
            try:
                entry = sub_q.get(timeout=1.0)
                _send({
                    "type": "session_event",
                    "username": username,
                    "session_id": session_id,
                    "entry": entry,
                })
            except queue.Empty:
                continue

    def _subscribe(username: str, session_id: str) -> None:
        key = f"{username}::{session_id}"
        with _lock:
            if key in _subscriptions:
                return
            sub_q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=500)
            _subscriptions[key] = sub_q
        register_history_subscriber(username=username, session_id=session_id, subscriber=sub_q)
        t = threading.Thread(
            target=_event_pump,
            args=(key, username, session_id, sub_q),
            daemon=True,
        )
        t.start()

    def _unsubscribe(username: str, session_id: str) -> None:
        key = f"{username}::{session_id}"
        with _lock:
            sub_q = _subscriptions.pop(key, None)
        if sub_q is not None:
            unregister_history_subscriber(username=username, session_id=session_id, subscriber=sub_q)

    def _unsubscribe_all() -> None:
        with _lock:
            pairs = list(_subscriptions.keys())
        for key in pairs:
            parts = key.split("::", 1)
            if len(parts) == 2:
                _unsubscribe(parts[0], parts[1])

    # ----------------------------------------------------------------- handlers

    def _handle_auth(msg: dict[str, Any]) -> None:
        nonlocal _auth_context
        username = str(msg.get("username", "")).strip()
        password = str(msg.get("password", "")).strip()
        if not username or not password:
            _send({"type": "auth_error", "message": "username_and_password_required"})
            return
        ctx = verify_user_password(runtime_root, username=username, password=password)
        if not ctx:
            _send({"type": "auth_error", "message": "invalid_credentials"})
            return
        _auth_context = ctx
        peer_meta = manifest.get("peer") or {}
        write_jsonl(log_path, {
            "type": "ws_peer.auth_ok",
            "ts": utc_ts(),
            "service_id": self_service["service_id"],
            "peer_username": username,
        })
        _send({
            "type": "auth_ok",
            "node_id": str(manifest.get("node_id") or ""),
            "peer_id": str(peer_meta.get("peer_id") or ""),
            "run_id": str(manifest.get("run_id") or ""),
        })

    def _handle_list_open_sessions(_msg: dict[str, Any]) -> None:
        if _auth_context is None:
            _send({"type": "error", "message": "not_authenticated"})
            return
        sessions = list_peer_joinable_sessions(runtime_root)
        _send({"type": "open_sessions", "sessions": sessions})

    def _handle_join_session(msg: dict[str, Any]) -> None:
        if _auth_context is None:
            _send({"type": "error", "message": "not_authenticated"})
            return
        username = str(msg.get("username", "")).strip()
        session_id = str(msg.get("session_id", "")).strip()
        if not username or not session_id:
            _send({"type": "error", "message": "username_and_session_id_required"})
            return
        # Verify the session is marked as peer-joinable
        open_sessions = list_peer_joinable_sessions(runtime_root)
        joinable = any(
            str(s.get("username", "")) == username and str(s.get("session_id", "")) == session_id
            for s in open_sessions
        )
        if not joinable:
            _send({"type": "error", "message": "session_not_joinable"})
            return
        _subscribe(username, session_id)
        write_jsonl(log_path, {
            "type": "ws_peer.session_joined",
            "ts": utc_ts(),
            "service_id": self_service["service_id"],
            "peer_username": str(_auth_context.get("username", "")),
            "target_username": username,
            "session_id": session_id,
        })
        _send({"type": "session_joined", "username": username, "session_id": session_id})

    def _handle_leave_session(msg: dict[str, Any]) -> None:
        username = str(msg.get("username", "")).strip()
        session_id = str(msg.get("session_id", "")).strip()
        _unsubscribe(username, session_id)
        _send({"type": "session_left", "username": username, "session_id": session_id})

    def _handle_message(msg: dict[str, Any]) -> None:
        if _auth_context is None:
            _send({"type": "error", "message": "not_authenticated"})
            return
        username = str(msg.get("username", "")).strip()
        session_id = str(msg.get("session_id", "")).strip()
        text = str(msg.get("text", "")).strip()
        if not username or not session_id:
            _send({"type": "error", "message": "username_and_session_id_required"})
            return
        if not text:
            _send({"type": "error", "message": "text_required"})
            return
        key = f"{username}::{session_id}"
        with _lock:
            if key not in _subscriptions:
                _send({"type": "error", "message": "not_joined_to_session"})
                return

        peer_label = str(_auth_context.get("username", "peer")).strip()
        full_text = f"[peer:{peer_label}] {text}"

        # Resolve or lease a service for the session
        leased = get_session_service(runtime_root, username=username, session_id=session_id)
        if not leased:
            pool = codex_service_pool if default_provider == "codex" else claude_service_pool
            if pool:
                leased = lease_session_service(
                    runtime_root,
                    username=username,
                    session_id=session_id,
                    pool_service_ids=pool,
                )

        append_history(
            username,
            session_id,
            {
                "direction": "out",
                "ts": utc_ts(),
                "to": leased or f"pending:{default_provider}",
                "session_id": session_id,
                "text": full_text,
                "peer_source": peer_label,
            },
        )
        append_pending_input(
            runtime_root,
            username=username,
            session_id=session_id,
            entry=make_aize_pending_input(kind="user_message", role="user", text=full_text),
        )

        if leased:
            ac = issue_auth_context(runtime_root, username=username)
            send_router_control(
                make_dispatch_pending_message(
                    manifest=manifest,
                    from_service_id=self_service["service_id"],
                    to_service_id=leased,
                    process_id=process_id,
                    run_id=manifest["run_id"],
                    username=username,
                    session_id=session_id,
                    auth_context=ac,
                    reason="ws_peer_message",
                )
            )
        write_jsonl(log_path, {
            "type": "ws_peer.message_dispatched",
            "ts": utc_ts(),
            "service_id": self_service["service_id"],
            "peer_username": peer_label,
            "target_username": username,
            "session_id": session_id,
            "to_service": leased,
        })
        _send({"type": "message_accepted", "username": username, "session_id": session_id})

    # ----------------------------------------------------------------- main loop

    _DISPATCH: dict[str, Callable[[dict[str, Any]], None]] = {
        "auth": _handle_auth,
        "list_open_sessions": _handle_list_open_sessions,
        "join_session": _handle_join_session,
        "leave_session": _handle_leave_session,
        "message": _handle_message,
    }

    try:
        while not _closed.is_set():
            result = read_frame(rfile)
            if result is None:
                break
            opcode, payload = result

            if opcode == OP_CLOSE:
                try:
                    write_close_frame(wfile)
                except Exception:
                    pass
                break

            if opcode == OP_PING:
                try:
                    write_pong_frame(wfile, payload)
                except Exception:
                    pass
                continue

            if opcode != OP_TEXT:
                # Ignore binary / continuation frames
                continue

            try:
                msg = json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                _send({"type": "error", "message": "invalid_json"})
                continue

            if not isinstance(msg, dict):
                _send({"type": "error", "message": "expected_object"})
                continue

            msg_type = str(msg.get("type", "")).strip()

            if msg_type == "ping":
                _send({"type": "pong"})
                continue

            handler = _DISPATCH.get(msg_type)
            if handler is None:
                _send({"type": "error", "message": f"unknown_type:{msg_type}"})
            else:
                handler(msg)

    finally:
        _closed.set()
        _unsubscribe_all()
