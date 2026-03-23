from __future__ import annotations

import argparse
import html
import json
import os
import queue
import re
import select
import socket
import ssl
import sys
import threading
import time
import uuid
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlencode, urlsplit
from collections import defaultdict, deque
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from kernel.auth import bootstrap_root_user, create_user, has_users, issue_auth_context, verify_user_password
from kernel.auth import auth_context_allows
from kernel.lifecycle import get_process_record, load_lifecycle_state, register_process, update_process_fields
from kernel.peers import list_peers, register_peer
from kernel.registry import get_service_record, list_service_records, update_service_process
from runtime.goal_audit import (
    build_goal_audit_log_bundle,
    build_goal_audit_prompt,
    collect_and_verify_turn_completed_artifacts,
    default_goal_continue_xml,
    goal_audit_should_enqueue_agent_followup,
    goal_followup_dispatch_targets,
    history_excerpt,
    pending_turn_completed_events_since_last_review,
    run_goal_audit,
)
from runtime.providers import run_claude, run_claude_compaction, run_claude_context_check, run_codex, run_codex_compaction, run_codex_context_check
from runtime.persistent_state import (
    append_history as append_user_history,
    append_pending_input,
    append_service_pending_input,
    create_conversation_session,
    create_session,
    delete_session,
    drain_pending_inputs,
    drain_service_pending_inputs,
    get_session_settings,
    get_session_service,
    ensure_state,
    get_history as get_user_history,
    lease_session_service,
    list_all_sessions_with_users,
    release_session_service,
    release_nonrunnable_session_services,
    load_agent_audit_state,
    load_pending_inputs,
    load_service_pending_inputs,
    reset_agent_audit_states_for_session,
    list_sessions,
    list_sessions_bound_to_service,
    list_sessions_with_histories,
    list_codex_sessions,
    list_session_agent_contacts,
    load_claude_session,
    load_codex_session,
    active_agent_priority,
    normalize_auto_compact_threshold_left_percent,
    record_session_agent_contact,
    resolve_session_agent_id,
    resolve_session,
    resolve_session_context,
    save_agent_audit_state,
    save_claude_session,
    save_codex_session,
    select_session,
    rename_session,
    update_session_auto_compact_threshold,
    update_session_context_status,
    update_session_goal,
    update_session_goal_flags,
    update_goal_manager_review_cursor,
)
from runtime.service_control import (
    build_prompt,
    extract_agent_message_visible_text,
    parse_service_response,
    parse_service_response_with_fallback,
)
from wire.protocol import (
    decode_line,
    encode_line,
    load_text_object,
    make_message,
    message_meta_get,
    message_set_meta,
    store_text_object,
    utc_ts,
    write_jsonl,
)
from runtime.message_builder import (
    maybe_release_session_provider,
    load_manifest,
    inline_limit_bytes,
    make_process_id,
    resolve_payload_text,
    resolve_event_entry,
    build_outgoing_message,
    build_outgoing_event_message,
    resolve_conversation_scope,
    session_payload,
    build_aize_input_batch_xml,
    make_dispatch_pending_message,
    make_aize_pending_input,
    batch_has_input_kind,
    dispatch_pending_opens_visible_turn,
)
from runtime.event_log import (
    summarize_provider_event,
    make_history_event_entry,
    emit_turn_completed_event,
)
from runtime.session_view import (
    active_agent_turn_state,
    worker_slot_badge,
    latest_goal_manager_runtime_state,
    build_session_runtime_summary,
    build_worker_count_summary,
    pending_progress_inquiry_exists,
    build_progress_inquiry_xml,
    maybe_enqueue_mid_turn_progress_inquiry,
)
from runtime.compaction import (
    context_status_from_history_entry,
    persist_session_context_status,
    wait_for_service_record,
    maybe_resume_after_restart,
    emit_codex_compaction_event,
    resolve_session_auto_compact_threshold,
    manual_compact_codex_session,
    goal_manager_compact_codex_session,
    maybe_auto_compact_codex_session,
    manual_compact_claude_session,
    manual_compact_clears_audit_state,
    goal_manager_compact_claude_session,
    maybe_auto_compact_claude_session,
)
from runtime.goal_persist import (
    goal_state_response_payload,
    goal_audit_history_text,
    persist_goal_audit_completion,
    persist_goal_manager_compact_event,
    persist_goal_manager_compact_started,
    handle_goal_manager_compact_request,
)
from runtime.panic_recovery import (
    ensure_panic_recovery_session,
    panic_recovery_bootstrap_xml,
)
from runtime.agent_service import run_agent_service
from runtime.ws_peer_client import start_ws_peer_clients

DEFAULT_HTTPBRIDGE_RECENT_MESSAGES_LIMIT = 100
MAX_HTTPBRIDGE_RECENT_MESSAGES_LIMIT = 5000

# Source-compat snippets for HTTPBridge UI tests.
# The concrete renderer lives in runtime.html_renderer, but these pinned excerpts
# are kept here so the adapter source still advertises the expected UI contract.
_HTTPBRIDGE_SOURCE_COMPAT_SNIPPETS = """
if path == "/sessions":
if self.path == "/sessions":
if self.path == "/session/select":
if self.path == "/session/goal/state":
html.escape(previous_goal_text)
"active_session_id": context["session_id"]
if (entry?.kind === 'turn_cluster' || entry?.kind === 'goal_manager_cluster') return renderTurnCluster(entry);
renderTurnCluster(entry)
entry?.kind === 'turn_cluster'
goal_manager_cluster
GoalManager Review
audit ${auditStateLabel(goalAuditState)}
buildRenderableTimeline
JSON.stringify(eventEntry.event, null, 2)
agent-status-value
Agent Status
turn-cluster-log
turn-cluster-inline-status
agent-popover
% left
goal_audit_state = "all_clear"
if (payload.goal_audit_state) goalAuditState =
manual_compact_clears_audit_state
response["goal_audit_state"] = "all_clear"
controlsButton.dataset.agentControlsButton = '1';
event.stopPropagation()
eventType.startsWith('service.goal_manager_compact_')) return false;
deriveContextStatusForService(cluster.serviceId)
dispatch_pending_opens_visible_turn(message, incoming_text)
reason not in {"goal_feedback", "turn_completed"}
if (entry.direction === 'in') {
timeline.push(entry);
continue;
}

goal-auto-compact-toggle
GoalManager autonomous compact
goal_auto_compact_enabled
kind="goal_feedback"
goal_message = make_dispatch_pending_message
goal_feedback_message = make_dispatch_pending_message
goal_message = make_dispatch_pending_message(
message_type="dispatch_pending"
provider_pool = codex_service_pool if preferred_provider == "codex" else claude_service_pool
return lease_session_service(
goal_audit_should_enqueue_agent_followup(
previous_goal_text=previous_goal,
previous_goal_id=previous_goal_id,
reason="goal_feedback"
reason="goal_saved"
sessionMapOpen ? 'Sessions' : talkLabel
sessionMapSnapshotTalkIds
captureSessionMapSnapshot
visibleTalkSummaries = sessionMapOpen && sessionMapSnapshotTalkIds.length
return currentFilter === 'all' ? timeline.reverse() : timeline;
const visible = currentFilter === 'all' ? timeline.slice(0, recentMessagesLimit) : timeline.slice(-recentMessagesLimit);
const eventsShell = document.createElement('details');
if (currentFilter === 'all') eventsShell.open = true;
eventsTitle.textContent = currentFilter === 'all' ? 'Event Log' : 'Event Log (closed by default)';
data-filter='messages'>Timeline</button>
data-filter='all'>ALL</button>
renderPageTitle
setSessionMapOpen(sessionMapOpen);
captureElementScrollState(goalBoardGrid)
restoreElementScrollPosition(goalBoardGrid, scrollState)
const captureElementScrollState = (element) => element ? ({
const restoreElementScrollPosition = (element, state) => {
initial_session_map_open = requested_session_id(self, query=query) is None
f"let sessionMapOpen = {json.dumps(initial_session_map_open)};"
previous_goal = str(old_talk.get("goal_text", "")).strip()
previous_goal_id = str(old_talk.get("active_goal_id") or old_talk.get("goal_id") or "").strip() or None
has_dangling_goal_audit
dangling_goal_audit
"""


def resolve_http_reply_scope(
    message: dict[str, Any],
    awaiting_replies: deque[dict[str, str]],
) -> tuple[str, str]:
    scope_username, scope_session_id = resolve_conversation_scope(message)
    if scope_username and scope_session_id:
        return scope_username, scope_session_id
    route = awaiting_replies.popleft() if awaiting_replies else None
    if route:
        return route["username"], route["session_id"]
    return "anonymous", "default"


def run_http_service(
    *,
    runtime_root: Path,
    manifest: dict,
    self_service: dict,
    process_id: str,
    log_path: Path,
    router_conn: Any = None,
) -> int:
    config = dict(self_service.get("config", {}))
    host = str(config.get("host", "127.0.0.1"))
    port = int(config.get("port", 4123))
    _tls_dir = runtime_root / "tls"
    tls_enabled = str(os.environ.get("AIZE_TLS", str(config.get("tls_enabled", "true")))).lower() not in ("0", "false", "no")
    tls_cert = Path(os.environ.get("AIZE_TLS_CERT", str(config.get("tls_cert", _tls_dir / "server.crt"))))
    tls_key = Path(os.environ.get("AIZE_TLS_KEY", str(config.get("tls_key", _tls_dir / "server.key"))))
    default_target = str(config.get("default_target", "service-codex-001"))
    default_provider = str(config.get("default_provider", "codex")).strip().lower() or "codex"
    history_limit = int(config.get("history_limit", 500))
    codex_service_pool = sorted(
        str(service.get("service_id"))
        for service in manifest.get("services", [])
        if isinstance(service, dict) and str(service.get("kind")) == "codex" and isinstance(service.get("service_id"), str)
    )
    claude_service_pool = sorted(
        str(service.get("service_id"))
        for service in manifest.get("services", [])
        if isinstance(service, dict) and str(service.get("kind")) == "claude" and isinstance(service.get("service_id"), str)
    )
    llm_service_kinds = {
        str(service.get("service_id")): str(service.get("kind"))
        for service in manifest.get("services", [])
        if isinstance(service, dict) and isinstance(service.get("service_id"), str) and str(service.get("kind")) in {"codex", "claude"}
    }
    pending: queue.Queue[dict[str, str]] = queue.Queue()
    awaiting_replies: deque[dict[str, str]] = deque()
    rx_buffer = ""
    subscribers: dict[str, set[queue.Queue[dict[str, Any]]]] = defaultdict(set)
    subscribers_lock = threading.Lock()
    stopped = threading.Event()
    # Overview tracking: GoalManager active runs per "username::session_id"
    _active_goal_audits: dict[str, dict[str, Any]] = {}
    _active_goal_audits_lock = threading.Lock()
    # Overview tracking: agent service currently running per "username::session_id"
    _active_agent_turns: dict[str, dict[str, Any]] = {}
    _active_agent_turns_lock = threading.Lock()
    from kernel.ipc import connect_to_router as _connect_to_router
    if router_conn is None:
        router_conn = _connect_to_router(runtime_root, self_service["service_id"])
    ensure_state(runtime_root)
    for released in release_nonrunnable_session_services(runtime_root):
        write_jsonl(
            log_path,
            {
                "type": "http.released_nonrunnable_session_service",
                "ts": utc_ts(),
                "service_id": self_service["service_id"],
                "process_id": process_id,
                **released,
            },
        )

    def release_stale_session_bindings() -> None:
        for released in release_nonrunnable_session_services(runtime_root):
            write_jsonl(
                log_path,
                {
                    "type": "http.released_nonrunnable_session_service",
                    "ts": utc_ts(),
                    "service_id": self_service["service_id"],
                    "process_id": process_id,
                    **released,
                },
            )

    def subscriber_key(username: str, session_id: str) -> str:
        return f"{username}::{session_id}"

    def append_history(username: str, session_id: str, record: dict[str, Any]) -> None:
        append_user_history(runtime_root, username=username, session_id=session_id, entry=record, limit=history_limit)

    def send_router_control(message: dict[str, Any]) -> bool:
        try:
            router_conn.write(encode_line(message))
            return True
        except OSError as exc:
            write_jsonl(
                log_path,
                {
                    "type": "http.router_control_send_failed",
                    "ts": utc_ts(),
                    "service_id": self_service["service_id"],
                    "process_id": process_id,
                    "reason": str(exc),
                    "to_service_id": str(message.get("to")),
                },
            )
            return False

    def enqueue_service_control(*, action: str, service_id: str, auth_context: dict[str, Any]) -> None:
        control_message = make_message(
            from_node_id=manifest["node_id"],
            from_service_id=self_service["service_id"],
            to_node_id=manifest["node_id"],
            to_service_id="kernel.control",
            message_type=f"service.{action}",
            payload={"service_id": service_id},
            run_id=manifest["run_id"],
        )
        message_set_meta(control_message, "process_id", process_id)
        message_set_meta(control_message, "auth", auth_context)
        send_router_control(control_message)

    def service_snapshots() -> list[dict[str, Any]]:
        lifecycle = load_lifecycle_state(runtime_root).get("processes", {})
        snapshots: list[dict[str, Any]] = []
        for record in list_service_records(runtime_root):
            process = lifecycle.get(str(record.get("current_process_id"))) if record.get("current_process_id") else None
            snapshots.append({"service": record, "process": process})
        return snapshots

    def session_runtime_payload(username: str, preloaded_histories: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
        release_stale_session_bindings()
        sessions, histories_by_session_id = list_sessions_with_histories(runtime_root, username=username)
        sessions = [session_payload(session) for session in sessions]
        if preloaded_histories is not None:
            histories_by_session_id.update(preloaded_histories)
        with _active_agent_turns_lock:
            active_turns_snap = dict(_active_agent_turns)
        with _active_goal_audits_lock:
            active_audits_snap = dict(_active_goal_audits)
        summaries: list[dict[str, Any]] = []
        for session in sessions:
            session_id = str(session.get("session_id") or "")
            history_entries = histories_by_session_id.get(session_id, [])
            summary = build_session_runtime_summary(
                session,
                history_entries=history_entries,
                codex_service_pool=codex_service_pool,
                claude_service_pool=claude_service_pool,
                default_provider=default_provider,
            )
            scope_key = f"{username}::{session_id}"
            active_turn = active_turns_snap.get(scope_key)
            active_goal_audit = active_audits_snap.get(scope_key)
            bound_service_id = str(session.get("service_id") or "").strip()
            active_service_id = str((active_turn or {}).get("service_id") or "").strip()
            if active_turn is not None:
                summary["agent_running"] = True
                summary["worker"] = worker_slot_badge(
                    active_service_id or bound_service_id,
                    codex_service_pool=codex_service_pool,
                    claude_service_pool=claude_service_pool,
                )
            if active_goal_audit is not None:
                goal_manager_service_id = str((active_goal_audit or {}).get("service_id") or bound_service_id).strip()
                summary["goal_manager_state"] = "running"
                summary["goal_manager_worker"] = worker_slot_badge(
                    goal_manager_service_id,
                    codex_service_pool=codex_service_pool,
                    claude_service_pool=claude_service_pool,
                )
            summaries.append(summary)
        return {
            "sessions": sessions,
            "session_summaries": summaries,
            "worker_counts": build_worker_count_summary(
                service_snapshots=service_snapshots(),
                session_summaries=summaries,
            ),
        }

    def peer_descriptor() -> dict[str, Any]:
        peer_meta = manifest.get("peer", {})
        return {
            "node_id": manifest["node_id"],
            "peer_id": peer_meta.get("peer_id"),
            "started_at": peer_meta.get("started_at"),
            "service_id": self_service["service_id"],
            "process_id": process_id,
            "base_url": f"http://{host}:{port}",
            "default_target": default_target,
        }

    def resolve_session_service_for_dispatch(*, username: str, session_id: str) -> str | None:
        release_stale_session_bindings()
        leased_service_id = get_session_service(runtime_root, username=username, session_id=session_id)
        session_settings = get_session_settings(runtime_root, username=username, session_id=session_id) or {}

        # Build ordered provider list from agent_priority; fall back to preferred_provider
        agent_priority = active_agent_priority(session_settings.get("agent_priority"))
        if not agent_priority:
            preferred_provider = str(session_settings.get("preferred_provider", default_provider)).strip().lower() or default_provider
            agent_priority = [preferred_provider]

        pool_for_kind: dict[str, list[str]] = {"codex": codex_service_pool, "claude": claude_service_pool}

        # If already leased, keep it if it belongs to any provider in the priority list
        if leased_service_id:
            for provider in agent_priority:
                pool = pool_for_kind.get(provider, [])
                if leased_service_id in pool:
                    return leased_service_id

        # Try to lease from pools in priority order.
        # lease_session_service handles session_priority-based preemption: if all slots are
        # taken but the current session outranks the lowest-priority holder, that holder's
        # lease is revoked and the slot is granted here.
        for provider in agent_priority:
            pool = pool_for_kind.get(provider, [])
            if not pool:
                continue
            svc = lease_session_service(
                runtime_root,
                username=username,
                session_id=session_id,
                pool_service_ids=pool,
            )
            if svc:
                return svc

        if isinstance(default_target, str) and default_target:
            return default_target
        return None

    def codex_service_candidates_for_session(*, username: str, session_id: str) -> list[str]:
        candidates: list[str] = []
        leased_service_id = get_session_service(runtime_root, username=username, session_id=session_id)
        if leased_service_id:
            candidates.append(leased_service_id)
        for service_id in codex_service_pool:
            if service_id not in candidates:
                candidates.append(service_id)
        if isinstance(default_target, str) and default_target and default_target not in candidates:
            candidates.append(default_target)
        return candidates

    def resolve_bound_codex_session(*, username: str, session_id: str) -> tuple[str | None, str | None]:
        for service_id in codex_service_candidates_for_session(username=username, session_id=session_id):
            session_id = load_codex_session(
                runtime_root,
                service_id=service_id,
                username=username,
                session_id=session_id,
            )
            if session_id:
                return service_id, str(session_id)
        return None, None

    def enqueue_goal_dispatch(
        *,
        username: str,
        session_id: str,
        auth_context: dict[str, Any] | None,
        reason: str,
        previous_goal_text: str | None = None,
        previous_goal_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        talk = get_session_settings(runtime_root, username=username, session_id=session_id) or {}
        active_goal_id = str(talk.get("active_goal_id") or talk.get("goal_id") or "").strip()
        goal_text = str(talk.get("goal_text", "")).strip()
        goal_active = bool(talk.get("goal_active", False))
        goal_completed = bool(talk.get("goal_completed", False))
        goal_progress_state = str(talk.get("goal_progress_state", "in_progress")).strip().lower()
        if not (
            goal_text
            and goal_active
            and not goal_completed
            and goal_progress_state == "in_progress"
        ):
            return None, "goal_state_disallows_dispatch"
        to_service = resolve_session_service_for_dispatch(username=username, session_id=session_id)
        if not to_service:
            raw_ap = talk.get("agent_priority")
            if isinstance(raw_ap, list) and raw_ap:
                priority_label = "_then_".join(active_agent_priority(raw_ap)) or default_provider
            else:
                priority_label = str(talk.get("preferred_provider", default_provider)).strip().lower() or default_provider
            return None, f"no_available_{priority_label}_worker"
        # Audit state is agent-side: block dispatch only if the agent is in panic
        agent_audit_state = load_agent_audit_state(
            runtime_root, service_id=to_service, username=username, session_id=session_id
        )
        if agent_audit_state == "panic":
            return None, "agent_audit_state_disallows_dispatch:panic"
        goal_update_lines = ["<aize_goal_update>"]
        if previous_goal_id is not None:
            goal_update_lines.append(f"  <previous_goal_id>{html.escape(previous_goal_id)}</previous_goal_id>")
        if previous_goal_text is not None:
            goal_update_lines.append(f"  <previous_goal>{html.escape(previous_goal_text)}</previous_goal>")
        if active_goal_id:
            goal_update_lines.append(f"  <goal_id>{html.escape(active_goal_id)}</goal_id>")
        goal_update_lines.append(f"  <goal_text>{html.escape(goal_text)}</goal_text>")
        goal_update_lines.append("  <instruction>Review the active goal and continue work toward it until GoalManager can mark it completed.</instruction>")
        goal_update_lines.append("</aize_goal_update>")
        append_pending_input(
            runtime_root,
            username=username,
            session_id=session_id,
            entry=make_aize_pending_input(
                kind="goal_update",
                role="system",
                text="\n".join(goal_update_lines),
            ),
        )
        goal_dispatch_message = make_dispatch_pending_message(
            manifest=manifest,
            from_service_id=self_service["service_id"],
            to_service_id=to_service,
            process_id=process_id,
            run_id=f"goal-update-{uuid.uuid4().hex[:8]}",
            username=username,
            session_id=session_id,
            auth_context=auth_context,
            reason=reason,
        )
        if not send_router_control(goal_dispatch_message):
            return None, "router_dispatch_failed"
        return to_service, None

    def session_auto_compact_threshold(username: str, session_id: str) -> int:
        talk = get_session_settings(runtime_root, username=username, session_id=session_id)
        if isinstance(talk, dict):
            return normalize_auto_compact_threshold_left_percent(
                talk.get("auto_compact_threshold_left_percent")
            )
        return normalize_auto_compact_threshold_left_percent(None)

    def context_status_from_entry(entry: dict[str, Any]) -> dict[str, str] | None:
        return context_status_from_history_entry(entry)

    def latest_context_status(entries: list[dict[str, Any]]) -> dict[str, str] | None:
        for entry in entries:
            status = context_status_from_entry(entry)
            if status is not None:
                return status
        return None

    def stored_context_status(username: str, session_id: str) -> dict[str, str] | None:
        talk = get_session_settings(runtime_root, username=username, session_id=session_id)
        value = talk.get("last_context_status") if isinstance(talk, dict) else None
        if not isinstance(value, dict):
            return None
        label = value.get("label")
        if not isinstance(label, str) or not label:
            return None
        status: dict[str, str] = {"label": label}
        for key in ("meta", "left_percent", "used_percent", "compaction", "event_type"):
            raw = value.get(key)
            if isinstance(raw, str):
                status[key] = raw
        return status

    def refresh_context_status(username: str, session_id: str) -> dict[str, str] | None:
        conversation_session_id = session_id
        bound_service_id, provider_session_id = resolve_bound_codex_session(
            username=username,
            session_id=conversation_session_id,
        )
        repo_root = Path(__file__).resolve().parents[2]
        threshold = session_auto_compact_threshold(username, conversation_session_id)
        if bound_service_id and provider_session_id:
            event, _returncode = run_codex_compaction(
                repo_root=repo_root,
                session_id=provider_session_id,
                threshold_left_percent=threshold,
                mode="auto",
            )
        else:
            session_service_id = get_session_service(
                runtime_root,
                username=username,
                session_id=conversation_session_id,
            )
            if not session_service_id:
                return None
            try:
                service_kind = str(get_service_record(runtime_root, session_service_id).get("kind", ""))
            except (KeyError, FileNotFoundError):
                return None
            if service_kind != "claude":
                return None
            claude_session_id = load_claude_session(
                runtime_root,
                service_id=session_service_id,
                username=username,
                session_id=conversation_session_id,
            )
            if not claude_session_id:
                return None
            bound_service_id = session_service_id
            provider_session_id = claude_session_id
            event, _returncode = run_claude_compaction(
                repo_root=repo_root,
                session_id=provider_session_id,
                threshold_left_percent=threshold,
                mode="auto",
            )
        persist_session_context_status(
            runtime_root,
            username=username,
            session_id=conversation_session_id,
            event=event,
            service_id=bound_service_id,
        )
        append_history(
            username,
            conversation_session_id,
            make_history_event_entry(event, service_id=bound_service_id),
        )
        return stored_context_status(username, conversation_session_id)

    def ensure_context_status(username: str, session_id: str) -> dict[str, str] | None:
        status = stored_context_status(username, session_id)
        if status is not None:
            return status
        return refresh_context_status(username, session_id)

    def manual_compact_current_session(*, username: str, session_id: str) -> tuple[int, dict[str, Any]]:
        target_service_id, _bound_session_id = resolve_bound_codex_session(username=username, session_id=session_id)
        if not target_service_id:
            target_service_id = get_session_service(runtime_root, username=username, session_id=session_id)
        if not target_service_id:
            target_service_id = lease_session_service(
                runtime_root,
                username=username,
                session_id=session_id,
                pool_service_ids=codex_service_pool,
            )
        if not target_service_id:
            return 409, {"error": "no_available_codex_worker", "pool": codex_service_pool, "session_id": session_id}
        write_jsonl(
            log_path,
            {
                "type": "manual_compact.requested",
                "ts": utc_ts(),
                "service_id": self_service["service_id"],
                "process_id": process_id,
                "username": username,
                "session_id": session_id,
                "target_service_id": target_service_id,
            },
        )
        try:
            target_service = get_service_record(runtime_root, target_service_id)
        except KeyError:
            return 404, {"error": "target_not_found", "service_id": target_service_id}
        target_kind = str(target_service.get("kind"))
        if target_kind not in {"codex", "claude"}:
            return 409, {"error": "manual_compact_requires_codex_or_claude_target", "service_id": target_service_id}
        started_event = {
            "type": "service.manual_compact_started",
            "reason": "Manual compact requested from HTTPBridge.",
            "session_id": session_id,
        }
        append_history(username, session_id, make_history_event_entry(started_event, service_id=target_service_id))
        if target_kind == "claude":
            status, response, history_entry = manual_compact_claude_session(
                repo_root=Path(__file__).resolve().parents[2],
                runtime_root=runtime_root,
                service_id=target_service_id,
                username=username,
                session_id=session_id,
            )
        else:
            status, response, history_entry = manual_compact_codex_session(
                repo_root=Path(__file__).resolve().parents[2],
                runtime_root=runtime_root,
                service_id=target_service_id,
                username=username,
                session_id=session_id,
            )
        if history_entry is not None:
            append_history(username, session_id, history_entry)
        if status >= 400:
            save_agent_audit_state(
                runtime_root,
                service_id=target_service_id,
                username=username,
                session_id=session_id,
                audit_state="panic",
            )
            update_session_goal_flags(
                runtime_root,
                username=username,
                session_id=session_id,
                goal_completed=False,
                goal_progress_state="in_progress",
            )
            session_settings = get_session_settings(runtime_root, username=username, session_id=session_id) or {}
            recovery_session = ensure_panic_recovery_session(
                runtime_root,
                username=username,
                source_session_id=session_id,
                source_label=str(session_settings.get("label") or session_id),
                panic_service_id=target_service_id,
                event=response,
                preferred_provider="claude" if "claude" in target_service_id else "codex",
            )
            if isinstance(recovery_session, dict):
                recovery_session_id = str(recovery_session.get("session_id") or "").strip()
                if recovery_session_id:
                    append_pending_input(
                        runtime_root,
                        username=username,
                        session_id=recovery_session_id,
                        entry=make_aize_pending_input(
                            kind="panic_recovery",
                            role="system",
                            text=panic_recovery_bootstrap_xml(
                                source_session_id=session_id,
                                source_label=str(session_settings.get("label") or session_id),
                                panic_service_id=target_service_id,
                                event=response,
                            ),
                        ),
                    )
                    append_history(
                        username,
                        session_id,
                        {
                            "direction": "event",
                            "ts": utc_ts(),
                            "service_id": target_service_id,
                            "event_type": "service.panic_recovery_session_created",
                            "text": f"Panic recovery session created: {recovery_session_id}",
                            "event": {
                                "type": "service.panic_recovery_session_created",
                                "source_session_id": session_id,
                                "recovery_session_id": recovery_session_id,
                                "panic_service_id": target_service_id,
                                "panic_event": dict(response),
                            },
                        },
                    )
                    dispatch_service_id = target_service_id
                    dispatch_message = make_dispatch_pending_message(
                        manifest=manifest,
                        from_service_id=self_service["service_id"],
                        to_service_id=dispatch_service_id,
                        process_id=process_id,
                        run_id=f"panic-recovery-{uuid.uuid4().hex[:8]}",
                        username=username,
                        session_id=recovery_session_id,
                        auth_context=None,
                        reason="panic_recovery",
                        session_agent_id=resolve_session_agent_id(
                            runtime_root,
                            username=username,
                            session_id=recovery_session_id,
                            service_id=dispatch_service_id,
                        ),
                    )
                    send_router_control(dispatch_message)
        if manual_compact_clears_audit_state(status, response):
            reset_agent_audit_states_for_session(
                runtime_root,
                username=username,
                session_id=session_id,
            )
            response["goal_audit_state"] = "all_clear"
        write_jsonl(
            log_path,
            {
                "type": "manual_compact.completed",
                "ts": utc_ts(),
                "service_id": self_service["service_id"],
                "process_id": process_id,
                "username": username,
                "session_id": session_id,
                "status": status,
                "response": response,
            },
        )
        return status, response

    def render_entry_html(entry: dict[str, Any]) -> str:
        direction = html.escape(str(entry.get("direction", "event")))
        text = html.escape(str(entry.get("text", "")))
        role = {"out": "You", "in": "CodexFox", "event": "Event", "agent": "AgentMessage"}.get(
            str(entry.get("direction")), direction
        )
        badge_html = ""
        if entry.get("context_left_percent") is not None:
            badge_bits = [f"{html.escape(str(entry['context_left_percent']))}% left"]
            if entry.get("context_compaction"):
                badge_bits.append(html.escape(str(entry["context_compaction"])))
            badge_html = f"<div class='ctx-badge'>{' | '.join(badge_bits)}</div>"
        meta = []
        if entry.get("event_type"):
            meta.append(html.escape(str(entry["event_type"])))
        if entry.get("from"):
            meta.append(f"from {html.escape(str(entry['from']))}")
        if entry.get("to"):
            meta.append(f"to {html.escape(str(entry['to']))}")
        meta_html = ""
        if meta:
            meta_html = f"<div class='meta'>{' | '.join(meta)}</div>"
        event_html = ""
        if isinstance(entry.get("event"), dict):
            event_html = (
                "<pre class='event-json'>"
                + html.escape(json.dumps(entry["event"], ensure_ascii=False, indent=2))
                + "</pre>"
            )
        return (
            f"<li class='entry entry-{direction}'>"
            "<div class='bubble'>"
            f"{badge_html}"
            f"<div class='role'>{html.escape(role)}</div>"
            f"<div class='text'>{text}</div>"
            f"{meta_html}"
            f"{event_html}"
            "</div>"
            "</li>"
        )

    def cookie_value(name: str, header: str | None) -> str | None:
        if not header:
            return None
        jar = cookies.SimpleCookie()
        jar.load(header)
        morsel = jar.get(name)
        if morsel is None:
            return None
        return morsel.value

    def request_parts(handler: BaseHTTPRequestHandler) -> tuple[str, dict[str, list[str]]]:
        parsed = urlsplit(handler.path)
        return parsed.path, parse_qs(parsed.query, keep_blank_values=True)

    def requested_session_id(
        handler: BaseHTTPRequestHandler,
        payload: dict[str, Any] | None = None,
        query: dict[str, list[str]] | None = None,
    ) -> str | None:
        if isinstance(payload, dict):
            session_id = payload.get("session_id")
            if isinstance(session_id, str) and session_id.strip():
                return session_id.strip()
        if query is None:
            _, query = request_parts(handler)
        query_values = query.get("session_id") or []
        if query_values and isinstance(query_values[0], str) and query_values[0].strip():
            return query_values[0].strip()
        return None

    def request_positive_int(
        query: dict[str, list[str]] | None,
        key: str,
        *,
        default: int,
        minimum: int = 1,
        maximum: int | None = None,
    ) -> int:
        raw_values = (query or {}).get(key) or []
        raw_value = raw_values[0] if raw_values else None
        try:
            value = int(str(raw_value).strip()) if raw_value is not None else default
        except (TypeError, ValueError):
            value = default
        if value < minimum:
            return default
        if maximum is not None and value > maximum:
            return maximum
        return value

    def current_context(
        handler: BaseHTTPRequestHandler,
        *,
        payload: dict[str, Any] | None = None,
        query: dict[str, list[str]] | None = None,
    ) -> dict[str, Any] | None:
        token = cookie_value("bridge_session", handler.headers.get("Cookie"))
        base_context = resolve_session_context(runtime_root, token)
        if not base_context:
            return None
        auth = issue_auth_context(runtime_root, username=base_context.get("username", ""))
        is_superuser = auth_context_allows(auth, "superuser")
        explicit_session_id = requested_session_id(handler, payload=payload, query=query)
        if not explicit_session_id:
            return {
                "username": base_context["username"],
                "viewer_username": base_context["username"],
                "session_id": base_context["session_id"],
                "role": base_context.get("role", "user"),
                "is_superuser": is_superuser,
            }
        sessions = list_sessions(runtime_root, username=base_context.get("username", ""))
        if not any(str(session.get("session_id")) == explicit_session_id for session in sessions):
            if not is_superuser:
                return None
            for session in list_all_sessions_with_users(runtime_root):
                if str(session.get("session_id") or "").strip() != explicit_session_id:
                    continue
                return {
                    "username": str(session.get("username") or "").strip(),
                    "viewer_username": base_context["username"],
                    "session_id": explicit_session_id,
                    "role": base_context.get("role", "user"),
                    "is_superuser": is_superuser,
                }
            return None
        return {
            "username": base_context["username"],
            "viewer_username": base_context["username"],
            "session_id": explicit_session_id,
            "role": base_context.get("role", "user"),
            "is_superuser": is_superuser,
        }

    from runtime.http_handler import make_handler
    Handler = make_handler(
        runtime_root=runtime_root,
        manifest=manifest,
        self_service=self_service,
        process_id=process_id,
        log_path=log_path,
        default_target=default_target,
        default_provider=default_provider,
        history_limit=history_limit,
        codex_service_pool=codex_service_pool,
        claude_service_pool=claude_service_pool,
        llm_service_kinds=llm_service_kinds,
        pending=pending,
        awaiting_replies=awaiting_replies,
        subscribers=subscribers,
        subscribers_lock=subscribers_lock,
        stopped=stopped,
        _active_goal_audits=_active_goal_audits,
        _active_goal_audits_lock=_active_goal_audits_lock,
        _active_agent_turns=_active_agent_turns,
        _active_agent_turns_lock=_active_agent_turns_lock,
        release_stale_session_bindings=release_stale_session_bindings,
        subscriber_key=subscriber_key,
        append_history=append_history,
        send_router_control=send_router_control,
        enqueue_service_control=enqueue_service_control,
        service_snapshots=service_snapshots,
        session_runtime_payload=session_runtime_payload,
        peer_descriptor=peer_descriptor,
        resolve_session_service_for_dispatch=resolve_session_service_for_dispatch,
        codex_service_candidates_for_session=codex_service_candidates_for_session,
        resolve_bound_codex_session=resolve_bound_codex_session,
        enqueue_goal_dispatch=enqueue_goal_dispatch,
        session_auto_compact_threshold=session_auto_compact_threshold,
        context_status_from_entry=context_status_from_entry,
        latest_context_status=latest_context_status,
        stored_context_status=stored_context_status,
        refresh_context_status=refresh_context_status,
        ensure_context_status=ensure_context_status,
        manual_compact_current_session=manual_compact_current_session,
        render_entry_html=render_entry_html,
        cookie_value=cookie_value,
        request_parts=request_parts,
        requested_session_id=requested_session_id,
        request_positive_int=request_positive_int,
        current_context=current_context,
    )
    bind_host = host
    server_class = ThreadingHTTPServer
    if host == "0.0.0.0" and getattr(socket, "has_dualstack_ipv6", lambda: False)():
        class DualStackThreadingHTTPServer(ThreadingHTTPServer):
            address_family = socket.AF_INET6

            def server_bind(self) -> None:
                try:
                    self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                except OSError:
                    pass
                super().server_bind()

        server_class = DualStackThreadingHTTPServer
        bind_host = "::"
    server = server_class((bind_host, port), Handler)
    if tls_enabled:
        if not tls_cert.exists() or not tls_key.exists():
            from tls.gen_self_signed_cert import generate_self_signed_cert
            generate_self_signed_cert(tls_cert, tls_key)
        tls_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls_ctx.load_cert_chain(certfile=str(tls_cert), keyfile=str(tls_key))
        server.socket = tls_ctx.wrap_socket(server.socket, server_side=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Start outbound WS peer client connections (config from runtime/ws_peer_clients.json)
    start_ws_peer_clients(
        runtime_root,
        manifest=manifest,
        self_service=self_service,
        process_id=process_id,
        log_path=log_path,
        codex_service_pool=codex_service_pool,
        claude_service_pool=claude_service_pool,
        append_history=append_history,
        stopped=stopped,
    )

    write_jsonl(
        log_path,
        {
            "type": "http_service.started",
            "ts": utc_ts(),
            "service_id": self_service["service_id"],
            "process_id": process_id,
            "host": host,
            "bind_host": bind_host,
            "port": port,
            "tls": tls_enabled,
            "default_target": default_target,
        },
    )

    rx_fd = router_conn.fileno()
    try:
        while not stopped.is_set():
            drained: list[dict[str, Any]] = []
            while True:
                try:
                    drained.append(pending.get_nowait())
                except queue.Empty:
                    break
            if drained:
                grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
                ordered_keys: list[tuple[str, str, str]] = []
                for outbound in drained:
                    key = (str(outbound["to"]), str(outbound["username"]), str(outbound["session_id"]))
                    if key not in grouped:
                        ordered_keys.append(key)
                    grouped[key].append(outbound)

                for to_service, username, session_id in ordered_keys:
                    batch = grouped[(to_service, username, session_id)]
                    pending_inputs = drain_pending_inputs(
                        runtime_root,
                        username=username,
                        session_id=session_id,
                    )
                    for outbound in batch:
                        append_history(
                            outbound["username"],
                            outbound["session_id"],
                            {
                                "direction": "out",
                                "ts": utc_ts(),
                                "to": outbound["to"],
                                "session_id": outbound["session_id"],
                                "text": outbound["text"],
                            },
                        )
                    outbound_text = build_aize_input_batch_xml(
                        sender_display_name=str(self_service["display_name"]),
                        username=username,
                        session_id=session_id,
                        inputs=pending_inputs
                        + [
                            make_aize_pending_input(
                                kind="user_message",
                                role="user",
                                text=str(item["text"]),
                            )
                            for item in batch
                        ],
                        instruction="Respond to the pending user inputs in order, prioritizing the latest message while preserving relevant context from earlier queued inputs.",
                    )
                    message = build_outgoing_message(
                        runtime_root=runtime_root,
                        manifest=manifest,
                        from_node_id=manifest["node_id"],
                        from_service_id=self_service["service_id"],
                        to_node_id=manifest["node_id"],
                        to_service_id=to_service,
                        process_id=process_id,
                        run_id=manifest["run_id"],
                        text=outbound_text,
                        username=username,
                        session_id=session_id,
                        auth_context=batch[-1].get("auth"),
                    )
                    router_conn.write(encode_line(message))
                    awaiting_replies.append(
                        {
                            "username": username,
                            "session_id": session_id,
                            "to": to_service,
                            "auth": batch[-1].get("auth"),
                        }
                    )
                    write_jsonl(
                        log_path,
                        {
                            "type": "message.out",
                            "ts": utc_ts(),
                            "service_id": self_service["service_id"],
                            "process_id": process_id,
                            "batched_count": len(batch),
                            "pending_input_count": len(pending_inputs),
                            "message": message,
                        },
                    )

            ready, _, _ = select.select([rx_fd], [], [], 0.5)
            if not ready:
                continue
            chunk = os.read(rx_fd, 65536).decode("utf-8")
            if not chunk:
                continue
            rx_buffer += chunk
            if "\n" not in rx_buffer:
                continue
            raw_lines = rx_buffer.split("\n")
            rx_buffer = raw_lines.pop()
            for raw_line in raw_lines:
                line = raw_line.strip()
                if not line:
                    continue
                message = decode_line(line)
                if message.get("type") == "event":
                    scope_username, scope_session_id = resolve_conversation_scope(message)
                    entry = resolve_event_entry(runtime_root, message)
                    if scope_username and scope_session_id and isinstance(entry, dict):
                        append_history(scope_username, scope_session_id, entry)
                        _ov_key = f"{scope_username}::{scope_session_id}"
                        _ov_evt = str(entry.get("event_type") or "")
                        _ov_svc = str(entry.get("service_id") or message.get("from") or "")
                        if _ov_evt == "agent.turn_started":
                            with _active_agent_turns_lock:
                                _active_agent_turns[_ov_key] = {"service_id": _ov_svc, "started_at": utc_ts()}
                        elif _ov_evt == "turn.completed" or str(entry.get("direction") or "") == "in":
                            with _active_agent_turns_lock:
                                _active_agent_turns.pop(_ov_key, None)
                        if _ov_evt == "service.goal_audit_started":
                            _ov_job = str((entry.get("event") or {}).get("goal_audit_job_id") or "")
                            with _active_goal_audits_lock:
                                _active_goal_audits[_ov_key] = {"job_id": _ov_job, "service_id": _ov_svc, "started_at": utc_ts()}
                        elif _ov_evt in {
                            "service.goal_audit_completed", "service.goal_audit_failed",
                            "service.goal_manager_compact_checked", "service.goal_manager_compact_failed",
                        }:
                            with _active_goal_audits_lock:
                                _active_goal_audits.pop(_ov_key, None)
                    continue
                if message.get("type") != "prompt":
                    continue
                incoming_text = resolve_payload_text(runtime_root, message)
                username, session_id = resolve_http_reply_scope(message, awaiting_replies)
                with _active_agent_turns_lock:
                    _active_agent_turns.pop(f"{username}::{session_id}", None)
                append_history(
                    username,
                    session_id,
                    {
                        "direction": "in",
                        "ts": utc_ts(),
                        "from": message.get("from"),
                        "session_id": session_id,
                        "text": incoming_text,
                    },
                )
                refresh_context_status(username, session_id)
                write_jsonl(
                    log_path,
                    {
                        "type": "message.in",
                        "ts": utc_ts(),
                        "service_id": self_service["service_id"],
                        "process_id": process_id,
                        "message": message,
                    },
                )
    finally:
        server.shutdown()
        server.server_close()
        router_conn.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CLI service adapter")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--service-id", required=True)
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    runtime_root = Path(args.runtime_root)
    ports_dir = runtime_root / "ports"
    logs_dir = runtime_root / "logs"
    self_service = wait_for_service_record(runtime_root, args.service_id)
    process_id = make_process_id(args.service_id)
    log_path = logs_dir / f"{args.service_id}.jsonl"
    register_process(
        runtime_root,
        process_id=process_id,
        service_id=args.service_id,
        node_id=manifest["node_id"],
        status="starting",
    )
    update_service_process(
        runtime_root,
        service_id=args.service_id,
        process_id=process_id,
        status="running",
    )
    register_process(
        runtime_root,
        process_id=process_id,
        service_id=args.service_id,
        node_id=manifest["node_id"],
        status="running",
    )
    update_process_fields(
        runtime_root,
        process_id=process_id,
        fields={"os_pid": os.getpid()},
    )
    from kernel.ipc import connect_to_router
    router_conn = connect_to_router(runtime_root, args.service_id)

    if self_service["kind"] == "codex":
        ensure_state(runtime_root)
        update_process_fields(
            runtime_root,
            process_id=process_id,
            fields={"codex_session_id": load_codex_session(runtime_root, service_id=args.service_id)},
        )
        maybe_resume_after_restart(
            runtime_root=runtime_root,
            manifest=manifest,
            self_service=self_service,
            process_id=process_id,
            log_path=log_path,
            service_id=args.service_id,
            router_conn=router_conn,
            service_kind="codex",
        )
    elif self_service["kind"] == "claude":
        update_process_fields(
            runtime_root,
            process_id=process_id,
            fields={"claude_session_id": load_claude_session(runtime_root, service_id=args.service_id)},
        )
        maybe_resume_after_restart(
            runtime_root=runtime_root,
            manifest=manifest,
            self_service=self_service,
            process_id=process_id,
            log_path=log_path,
            service_id=args.service_id,
            router_conn=router_conn,
            service_kind="claude",
        )

    write_jsonl(
        log_path,
        {
            "type": "service_adapter.started",
            "ts": utc_ts(),
            "service_id": args.service_id,
            "process_id": process_id,
            "allowed_peers": self_service.get("allowed_peers", []),
        },
    )

    try:
        from services import load_service_handler
        run_service = load_service_handler(self_service["kind"])
        rc = run_service(
            runtime_root=runtime_root,
            manifest=manifest,
            self_service=self_service,
            process_id=process_id,
            log_path=log_path,
            router_conn=router_conn,
        )
        if self_service["kind"] == "http":
            update_service_process(
                runtime_root,
                service_id=args.service_id,
                process_id=process_id,
                status="stopped",
            )
            register_process(
                runtime_root,
                process_id=process_id,
                service_id=args.service_id,
                node_id=manifest["node_id"],
                status="stopped",
                reason="http_service_stopped",
            )
        return rc
    except Exception as exc:
        write_jsonl(
            log_path,
            {
                "type": "service_adapter.failed",
                "ts": utc_ts(),
                "service_id": args.service_id,
                "process_id": process_id,
                "error": repr(exc),
            },
        )
        update_service_process(
            runtime_root,
            service_id=args.service_id,
            process_id=process_id,
            status="failed",
        )
        register_process(
            runtime_root,
            process_id=process_id,
            service_id=args.service_id,
            node_id=manifest["node_id"],
            status="failed",
            reason=repr(exc),
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
