"""Outbound WS peer client — persistent AIze-to-AIze agent participation.

This module implements an outbound WebSocket client that connects a local Aize
instance to a remote Aize session as an external agent.  Multiple client
configurations can run concurrently, each in its own daemon thread.

Flow per connection
-------------------
1. Connect to remote_ws_url (wss://host:port/ws), TLS with cert-verify disabled
2. Send {"type":"auth","username":...,"password":...} → wait for auth_ok
3. Send {"type":"join_session","username":...,"session_id":...} → wait for session_joined
4. Start keepalive ping thread (every PING_INTERVAL seconds)
5. Main receive loop:
   - session_event with entry.direction == "out" → dispatch to local LLM → send message back
   - pong / error → log and continue
6. On any disconnect: exponential back-off reconnect, then restart from step 1

Config dict keys (from .agent-mesh-runtime/ws_peer_clients.json)
-----------------------------------------------------------------
enabled            bool   default True
target_ws_url      str    wss://host:port/path
auth_username      str    credential for the remote Aize
auth_password      str
remote_username    str    owner of the remote session to join
remote_session_id  str    session_id on the remote Aize
local_username     str    username for the local proxy session
local_session_id   str    session_id of the local proxy session (LLM runs here)
provider           str    "claude" | "codex"  (default "claude")
name               str    optional label used in log messages
"""
from __future__ import annotations

import json
import os
import queue
import socket
import ssl
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from runtime.ws_bridge import (
    OP_CLOSE,
    OP_PING,
    OP_TEXT,
    read_frame,
    write_close_frame,
    write_masked_text_frame,
    write_pong_frame,
)
from wire.protocol import encode_line, make_message, message_set_meta, utc_ts, write_jsonl

PING_INTERVAL = 30          # seconds between client-side pings
RESPONSE_TIMEOUT = 300      # seconds to wait for local LLM response
RECONNECT_BACKOFF_BASE = 5  # initial reconnect wait (seconds)
RECONNECT_BACKOFF_MAX = 120 # cap on reconnect wait


def _send(wfile: Any, msg: dict[str, Any], write_lock: threading.Lock) -> bool:
    try:
        with write_lock:
            write_masked_text_frame(wfile, json.dumps(msg, ensure_ascii=False))
        return True
    except Exception:
        return False


def _ws_upgrade(host: str, port: int, path: str, use_tls: bool) -> tuple[socket.socket, Any, Any]:
    """Open a TCP/TLS socket and perform the RFC 6455 HTTP upgrade.

    Returns (raw_socket, rfile, wfile).  The caller owns the socket and must
    close it when done.
    """
    import base64

    sock = socket.create_connection((host, port), timeout=15)
    if use_tls:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        sock = ctx.wrap_socket(sock, server_hostname=host)

    # Use a long read timeout so quiet periods don't drop the connection;
    # the keepalive ping thread sends a ping every PING_INTERVAL seconds so
    # we'll always see traffic within 2*PING_INTERVAL.
    sock.settimeout(max(PING_INTERVAL * 3, 120))

    # Make rfile/wfile from the socket
    rfile = sock.makefile("rb")
    wfile = sock.makefile("wb")

    ws_key = base64.b64encode(os.urandom(16)).decode("ascii")
    handshake = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {ws_key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    )
    wfile.write(handshake.encode("utf-8"))
    wfile.flush()

    # Read status line
    status = rfile.readline().decode("ascii", errors="replace").strip()
    if "101" not in status:
        raise ConnectionError(f"WS upgrade failed: {status!r}")

    # Drain remaining HTTP response headers
    while True:
        line = rfile.readline()
        if line in (b"\r\n", b"\n", b""):
            break

    # After upgrade, set a longer read timeout so the receive loop
    # survives idle periods between keepalive pings (PING_INTERVAL=30s).
    sock.settimeout(90)

    return sock, rfile, wfile


def _send_router_control(runtime_root: Path, message: dict[str, Any]) -> bool:
    control_port = runtime_root / "ports" / "router.control"
    payload = encode_line(message).encode("utf-8")
    fd: int | None = None
    try:
        fd = os.open(str(control_port), os.O_RDWR | os.O_NONBLOCK)
        os.write(fd, payload)
        return True
    except OSError:
        return False
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _dispatch_to_local_llm(
    *,
    runtime_root: Path,
    manifest: dict[str, Any],
    self_service: dict[str, Any],
    process_id: str,
    local_username: str,
    local_session_id: str,
    prompt_text: str,
    provider_pool: list[str],
    log_path: Path,
    append_history: Any,
    write_jsonl_fn: Any,
) -> str | None:
    """Dispatch prompt_text to the local LLM and return the response text.

    Returns the first ``direction="in"`` text seen in the local session history
    after the dispatch, or None on timeout / error.

    NOTE: The agent response is written by a different process (service-claude-*)
    so we poll the on-disk history rather than using in-process queue subscribers.
    """
    from kernel.auth import issue_auth_context
    from runtime.message_builder import make_aize_pending_input, make_dispatch_pending_message
    from runtime.persistent_state_pkg import (
        append_pending_input,
        get_history,
        get_session_service,
        lease_session_service,
    )

    auth_context = issue_auth_context(runtime_root, username=local_username)
    dispatch_ts = utc_ts()

    # Lease a local LLM service for the proxy session.
    leased_service_id = get_session_service(
        runtime_root, username=local_username, session_id=local_session_id
    )
    if leased_service_id and provider_pool and leased_service_id not in provider_pool:
        leased_service_id = None
    if not leased_service_id:
        leased_service_id = lease_session_service(
            runtime_root,
            username=local_username,
            session_id=local_session_id,
            pool_service_ids=provider_pool,
        )
    if not leased_service_id:
        write_jsonl_fn(log_path, {
            "type": "ws_peer_client.dispatch_failed",
            "ts": utc_ts(),
            "reason": "no_available_provider_worker",
            "local_session_id": local_session_id,
        })
        return None

    # Write the user message to local history and pending inputs.
    append_history(local_username, local_session_id, {
        "direction": "out",
        "ts": dispatch_ts,
        "to": leased_service_id,
        "session_id": local_session_id,
        "text": prompt_text,
    })
    append_pending_input(
        runtime_root,
        username=local_username,
        session_id=local_session_id,
        entry=make_aize_pending_input(
            kind="user_message",
            role="user",
            text=prompt_text,
        ),
    )

    # Send dispatch control message to the router.
    dispatch_msg = make_dispatch_pending_message(
        manifest=manifest,
        from_service_id=self_service["service_id"],
        to_service_id=leased_service_id,
        process_id=process_id,
        run_id=manifest["run_id"],
        username=local_username,
        session_id=local_session_id,
        auth_context=auth_context,
        reason="ws_peer_client_prompt",
    )
    if not _send_router_control(runtime_root, dispatch_msg):
        write_jsonl_fn(log_path, {
            "type": "ws_peer_client.dispatch_failed",
            "ts": utc_ts(),
            "reason": "router_control_send_failed",
            "local_session_id": local_session_id,
        })
        return None

    # Poll the on-disk session history for the response.
    # The agent response is written by a separate process (service-claude-*), so
    # in-process queue subscribers won't see it — we must read from disk.
    deadline = time.monotonic() + RESPONSE_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(2.0)
        try:
            history = get_history(
                runtime_root,
                username=local_username,
                session_id=local_session_id,
            )
        except Exception:
            continue
        for entry in history:
            if not isinstance(entry, dict):
                continue
            if entry.get("direction") != "in":
                continue
            if str(entry.get("session_id", "")) != local_session_id:
                continue
            entry_ts = str(entry.get("ts", ""))
            if entry_ts > dispatch_ts:
                return str(entry.get("text", "")).strip()
    # Timeout
    write_jsonl_fn(log_path, {
        "type": "ws_peer_client.response_timeout",
        "ts": utc_ts(),
        "local_session_id": local_session_id,
    })
    return None


def run_ws_peer_client(
    config: dict[str, Any],
    *,
    runtime_root: Path,
    manifest: dict[str, Any],
    self_service: dict[str, Any],
    process_id: str,
    log_path: Path,
    codex_service_pool: list[str],
    claude_service_pool: list[str],
    append_history: Any,
    stopped: threading.Event,
) -> None:
    """Run a single outbound WS peer client connection loop (blocking).

    Call this from a daemon thread.  It will reconnect automatically until
    *stopped* is set.
    """
    name = str(config.get("name") or "ws-peer-client")
    target_ws_url = str(config.get("target_ws_url", "")).strip()
    auth_username = str(config.get("auth_username", "")).strip()
    auth_password = str(config.get("auth_password", "")).strip()
    remote_username = str(config.get("remote_username", "")).strip()
    remote_session_id = str(config.get("remote_session_id", "")).strip()
    local_username = str(config.get("local_username", auth_username)).strip()
    local_session_id = str(config.get("local_session_id", "")).strip()
    provider = str(config.get("provider", "claude")).strip().lower()

    if not all([target_ws_url, auth_username, auth_password, remote_username, remote_session_id]):
        write_jsonl(log_path, {
            "type": "ws_peer_client.config_invalid",
            "ts": utc_ts(),
            "name": name,
            "reason": "missing_required_fields",
        })
        return

    provider_pool = codex_service_pool if provider == "codex" else claude_service_pool

    # Auto-create the local proxy session if not configured or not found.
    if not local_session_id:
        from runtime.persistent_state_pkg import (
            create_conversation_session,
            list_sessions,
            update_session_goal,
        )
        existing = list_sessions(runtime_root, username=local_username)
        proxy = next((s for s in existing if s.get("label") == f"[ws-proxy] {name}"), None)
        if proxy:
            local_session_id = str(proxy["session_id"])
        else:
            session = create_conversation_session(
                runtime_root,
                username=local_username,
                label=f"[ws-proxy] {name}",
            )
            if session:
                local_session_id = str(session["session_id"])
        # Ensure the proxy session has an active goal so the periodic
        # release_nonrunnable_session_services sweep doesn't evict its lease.
        if local_session_id:
            update_session_goal(
                runtime_root,
                username=local_username,
                session_id=local_session_id,
                goal_text=f"WS proxy agent for {name}",
            )
        write_jsonl(log_path, {
            "type": "ws_peer_client.proxy_session",
            "ts": utc_ts(),
            "name": name,
            "local_session_id": local_session_id,
        })

    parsed = urlparse(target_ws_url)
    use_tls = parsed.scheme in ("wss", "https")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if use_tls else 80)
    path = parsed.path or "/ws"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    backoff = RECONNECT_BACKOFF_BASE

    while not stopped.is_set():
        sock = None
        rfile = None
        wfile = None
        write_lock = threading.Lock()

        try:
            write_jsonl(log_path, {
                "type": "ws_peer_client.connecting",
                "ts": utc_ts(),
                "name": name,
                "target": target_ws_url,
            })
            sock, rfile, wfile = _ws_upgrade(host, port, path, use_tls)
            backoff = RECONNECT_BACKOFF_BASE  # reset on success

            # --- Authenticate ---
            _send(wfile, {
                "type": "auth",
                "username": auth_username,
                "password": auth_password,
            }, write_lock)

            # Wait for auth_ok (drain until we get it or error)
            auth_ok = False
            for _ in range(20):
                frame = read_frame(rfile)
                if frame is None:
                    raise ConnectionError("Connection closed during auth")
                opcode, payload = frame
                if opcode == OP_CLOSE:
                    raise ConnectionError("Server closed during auth")
                if opcode != OP_TEXT:
                    continue
                msg = json.loads(payload.decode("utf-8"))
                if msg.get("type") == "auth_ok":
                    auth_ok = True
                    break
                if msg.get("type") == "auth_error":
                    raise PermissionError(f"auth_error: {msg.get('message')}")
            if not auth_ok:
                raise ConnectionError("auth_ok not received")

            # --- Join session ---
            _send(wfile, {
                "type": "join_session",
                "username": remote_username,
                "session_id": remote_session_id,
            }, write_lock)

            # Wait for session_joined
            joined = False
            for _ in range(20):
                frame = read_frame(rfile)
                if frame is None:
                    raise ConnectionError("Connection closed during join")
                opcode, payload = frame
                if opcode == OP_CLOSE:
                    raise ConnectionError("Server closed during join")
                if opcode != OP_TEXT:
                    continue
                msg = json.loads(payload.decode("utf-8"))
                if msg.get("type") == "session_joined":
                    joined = True
                    break
                if msg.get("type") == "error":
                    raise RuntimeError(f"join error: {msg.get('message')}")
            if not joined:
                raise ConnectionError("session_joined not received")

            write_jsonl(log_path, {
                "type": "ws_peer_client.joined",
                "ts": utc_ts(),
                "name": name,
                "remote_username": remote_username,
                "remote_session_id": remote_session_id,
            })

            # --- Backlog recovery ---
            # Scan the remote session's recent history for "out" prompts that
            # were never answered by a WS peer (no subsequent direction="in"
            # from a ws-peer service_id).  Process them now so prompts sent
            # while the client was disconnected are not silently lost.
            def _recover_backlog() -> None:
                try:
                    from runtime.persistent_state_pkg import get_history
                    history = get_history(
                        runtime_root,
                        username=remote_username,
                        session_id=remote_session_id,
                    )
                    # Walk entries in chronological order (history is newest-first)
                    entries_asc = list(reversed(history))
                    # Collect "out" entries that have no following "in" from ws-peer
                    unanswered: list[str] = []
                    seen_out: list[str] = []
                    answered_ts: set[str] = set()
                    for e in entries_asc:
                        d = str(e.get("direction", ""))
                        frm = str(e.get("from", ""))
                        if d == "in" and frm.startswith("ws-peer"):
                            # Mark the most recent unanswered "out" as answered
                            if seen_out:
                                answered_ts.add(seen_out[-1])
                        elif d == "out":
                            seen_out.append(str(e.get("ts", "")))
                    unanswered_entries = [
                        e for e in entries_asc
                        if str(e.get("direction", "")) == "out"
                        and str(e.get("ts", "")) not in answered_ts
                        and str(e.get("text", "")).strip()
                    ]
                    # Only take the last 3 unanswered to avoid flooding
                    unanswered_entries = unanswered_entries[-3:]
                    if unanswered_entries:
                        write_jsonl(log_path, {
                            "type": "ws_peer_client.backlog_recovery",
                            "ts": utc_ts(),
                            "name": name,
                            "count": len(unanswered_entries),
                        })
                    for entry in unanswered_entries:
                        pt = str(entry.get("text", "")).strip()
                        if not pt:
                            continue
                        response_text = _dispatch_to_local_llm(
                            runtime_root=runtime_root,
                            manifest=manifest,
                            self_service=self_service,
                            process_id=process_id,
                            local_username=local_username,
                            local_session_id=local_session_id,
                            prompt_text=pt,
                            provider_pool=provider_pool,
                            log_path=log_path,
                            append_history=append_history,
                            write_jsonl_fn=write_jsonl,
                        )
                        if response_text:
                            _send(wfile, {
                                "type": "message",
                                "username": remote_username,
                                "session_id": remote_session_id,
                                "text": f"[backlog] {response_text}",
                            }, write_lock)
                except Exception as exc:
                    write_jsonl(log_path, {
                        "type": "ws_peer_client.backlog_error",
                        "ts": utc_ts(),
                        "name": name,
                        "error": repr(exc),
                    })

            threading.Thread(target=_recover_backlog, daemon=True).start()

            # --- Keepalive ping thread ---
            closed_event = threading.Event()

            def _ping_loop() -> None:
                while not stopped.is_set() and not closed_event.is_set():
                    time.sleep(PING_INTERVAL)
                    if closed_event.is_set():
                        break
                    if not _send(wfile, {"type": "ping"}, write_lock):
                        closed_event.set()

            ping_thread = threading.Thread(target=_ping_loop, daemon=True)
            ping_thread.start()

            # --- Main receive loop ---
            while not stopped.is_set() and not closed_event.is_set():
                try:
                    frame = read_frame(rfile)
                except TimeoutError:
                    # Socket read timed out (no frame for sock.settimeout seconds).
                    # This is not a real disconnect — the server-side keepalive
                    # is handled at a higher level; just retry the read.
                    continue
                if frame is None:
                    break
                opcode, payload = frame

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
                    continue

                try:
                    msg = json.loads(payload.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                if not isinstance(msg, dict):
                    continue

                msg_type = str(msg.get("type", ""))

                if msg_type == "pong":
                    continue

                if msg_type == "session_event":
                    entry = msg.get("entry", {})
                    if not isinstance(entry, dict):
                        continue
                    direction = str(entry.get("direction", ""))
                    # Only act on user-visible outbound prompts
                    if direction != "out":
                        continue
                    prompt_text = str(entry.get("text", "")).strip()
                    if not prompt_text:
                        continue

                    write_jsonl(log_path, {
                        "type": "ws_peer_client.prompt_received",
                        "ts": utc_ts(),
                        "name": name,
                        "remote_session_id": remote_session_id,
                        "text_preview": prompt_text[:120],
                    })

                    # Dispatch in a separate thread so the receive loop stays live.
                    def _handle_prompt(pt: str = prompt_text) -> None:
                        response_text = _dispatch_to_local_llm(
                            runtime_root=runtime_root,
                            manifest=manifest,
                            self_service=self_service,
                            process_id=process_id,
                            local_username=local_username,
                            local_session_id=local_session_id,
                            prompt_text=pt,
                            provider_pool=provider_pool,
                            log_path=log_path,
                            append_history=append_history,
                            write_jsonl_fn=write_jsonl,
                        )
                        if response_text:
                            _send(wfile, {
                                "type": "message",
                                "username": remote_username,
                                "session_id": remote_session_id,
                                "text": response_text,
                            }, write_lock)
                            write_jsonl(log_path, {
                                "type": "ws_peer_client.response_sent",
                                "ts": utc_ts(),
                                "name": name,
                                "text_preview": response_text[:120],
                            })

                    threading.Thread(target=_handle_prompt, daemon=True).start()
                    continue

                if msg_type == "error":
                    write_jsonl(log_path, {
                        "type": "ws_peer_client.remote_error",
                        "ts": utc_ts(),
                        "name": name,
                        "message": msg.get("message"),
                    })

            closed_event.set()

        except PermissionError:
            # Auth credentials wrong — don't reconnect, just stop this client
            write_jsonl(log_path, {
                "type": "ws_peer_client.auth_failed_stopping",
                "ts": utc_ts(),
                "name": name,
            })
            return

        except Exception as exc:
            write_jsonl(log_path, {
                "type": "ws_peer_client.disconnected",
                "ts": utc_ts(),
                "name": name,
                "error": repr(exc),
                "reconnect_in": backoff,
            })
        finally:
            try:
                if rfile:
                    rfile.close()
                if wfile:
                    wfile.close()
                if sock:
                    sock.close()
            except Exception:
                pass

        if stopped.is_set():
            break

        # Exponential backoff before reconnect
        for _ in range(int(backoff)):
            if stopped.is_set():
                return
            time.sleep(1)
        backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX)


def load_ws_peer_client_configs(runtime_root: Path) -> list[dict[str, Any]]:
    """Load outbound WS peer client configurations from runtime dir.

    Config file: <runtime_root>/ws_peer_clients.json
    Returns a list of config dicts; only those with enabled=True (default) are returned.
    """
    config_path = runtime_root / "ws_peer_clients.json"
    if not config_path.exists():
        return []
    try:
        configs = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(configs, list):
            return []
        return [c for c in configs if isinstance(c, dict) and c.get("enabled", True)]
    except Exception:
        return []


def start_ws_peer_clients(
    runtime_root: Path,
    *,
    manifest: dict[str, Any],
    self_service: dict[str, Any],
    process_id: str,
    log_path: Path,
    codex_service_pool: list[str],
    claude_service_pool: list[str],
    append_history: Any,
    stopped: threading.Event,
) -> list[threading.Thread]:
    """Start one daemon thread per enabled WS peer client config.

    Returns the list of started threads (all daemon threads).
    """
    configs = load_ws_peer_client_configs(runtime_root)
    threads: list[threading.Thread] = []
    for i, cfg in enumerate(configs):
        if not cfg.get("name"):
            cfg = dict(cfg)
            cfg["name"] = f"ws-peer-{i}"
        t = threading.Thread(
            target=run_ws_peer_client,
            kwargs=dict(
                config=cfg,
                runtime_root=runtime_root,
                manifest=manifest,
                self_service=self_service,
                process_id=process_id,
                log_path=log_path,
                codex_service_pool=codex_service_pool,
                claude_service_pool=claude_service_pool,
                append_history=append_history,
                stopped=stopped,
            ),
            name=f"ws-peer-client-{i}",
            daemon=True,
        )
        t.start()
        threads.append(t)
        write_jsonl(log_path, {
            "type": "ws_peer_client.started",
            "ts": utc_ts(),
            "name": cfg.get("name"),
            "target": cfg.get("target_ws_url"),
            "remote_session_id": cfg.get("remote_session_id"),
        })
    return threads
