from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from kernel.registry import get_service_record
from runtime.compaction import persist_session_context_status
from runtime.event_log import make_history_event_entry
from runtime.persistent_state import (
    append_history as append_user_history,
    append_jsonl,
    get_session_settings,
    read_json_file,
    save_agent_audit_state,
    session_goal_manager_reviews_path,
    session_goal_manager_state_path,
    update_goal_manager_review_cursor,
    update_session_goal_flags,
    write_json_file,
)
from wire.protocol import utc_ts, write_jsonl

GOAL_AUDIT_HISTORY_LIMIT = 500


def _persist_goal_manager_runtime_state(
    *,
    runtime_root: Path,
    username: str,
    session_id: str,
    payload: dict[str, Any],
) -> None:
    state_path = session_goal_manager_state_path(runtime_root, username=username, session_id=session_id)
    state = read_json_file(state_path) or {}
    state.update(payload)
    state["updated_at"] = utc_ts()
    write_json_file(state_path, state)


def goal_state_response_payload(
    talk: dict[str, Any],
    *,
    session_id: str,
    default_provider: str,
    agent_audit_state: str | None = None,
    goal_manager_state: str | None = None,
    goal_manager_service_id: str | None = None,
    goal_manager_worker: dict[str, Any] | None = None,
    welcomed_agents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    # audit_state is agent-side; use provided value or default to all_clear (session no longer stores it)
    resolved_audit_state = (
        agent_audit_state
        if isinstance(agent_audit_state, str) and agent_audit_state in {"all_clear", "needs_compact", "panic"}
        else "all_clear"
    )
    return {
        "ok": True,
        "session_id": session_id,
        "goal_id": str(talk.get("goal_id", "") or ""),
        "active_goal_id": str(talk.get("active_goal_id", "") or ""),
        "goal_text": str(talk.get("goal_text", "")),
        "goal_mode": str(talk.get("goal_mode", "no_goal")),
        "goal_active": bool(talk.get("goal_active", False)),
        "goal_completed": bool(talk.get("goal_completed", False)),
        "goal_progress_state": str(talk.get("goal_progress_state", "in_progress")),
        "goal_history": list(talk.get("goal_history", []))
        if isinstance(talk.get("goal_history"), list)
        else [],
        "goal_audit_state": resolved_audit_state,
        "goal_reset_completed_on_prompt": bool(talk.get("goal_reset_completed_on_prompt", True)),
        "goal_auto_compact_enabled": bool(talk.get("goal_auto_compact_enabled", True)),
        "agent_welcome_enabled": bool(talk.get("agent_welcome_enabled", False)),
        "preferred_provider": str(talk.get("preferred_provider", default_provider)),
        "session_group": str(talk.get("session_group", "user")),
        "auto_resume_enabled": bool(talk.get("auto_resume_enabled", False)),
        "auto_resume_interval_seconds": int(talk.get("auto_resume_interval_seconds", 21600) or 21600),
        "auto_resume_next_at": str(talk.get("auto_resume_next_at", "") or ""),
        "auto_resume_reason": str(talk.get("auto_resume_reason", "") or ""),
        "auto_resume_last_scheduled_at": str(talk.get("auto_resume_last_scheduled_at", "") or ""),
        "auto_resume_last_started_at": str(talk.get("auto_resume_last_started_at", "") or ""),
        "auto_resume_last_error": str(talk.get("auto_resume_last_error", "") or ""),
        "goal_manager_state": str(goal_manager_state or "idle"),
        "goal_manager_service_id": str(goal_manager_service_id or ""),
        "goal_manager_worker": goal_manager_worker if isinstance(goal_manager_worker, dict) else None,
        "welcomed_agents": welcomed_agents if welcomed_agents is not None else list(talk.get("welcomed_agents", [])) if isinstance(talk.get("welcomed_agents"), list) else [],
        "selected_agents": list(talk.get("selected_agents", [])) if isinstance(talk.get("selected_agents"), list) else [],
    }


def goal_audit_history_text(audit: dict[str, Any]) -> str:
    status = str(audit.get("progress_state", "complete" if audit["goal_satisfied"] else "in_progress"))
    audit_state = str(audit.get("audit_state", "all_clear"))
    compact_requested = bool(audit.get("request_compact"))
    compact_reason = str(audit.get("request_compact_reason", "")).strip()
    lines = [
        f"GoalManager session {audit['goal_audit_session_id'] or '(ephemeral)'}: progress={status}, audit={audit_state}",
    ]
    if audit.get("summary"):
        lines.extend(["", str(audit["summary"])])
    if compact_requested:
        lines.extend(["", f"Compact requested: {compact_reason or 'no reason provided'}"])
    return "\n".join(lines)


def persist_goal_audit_completion(
    *,
    runtime_root: Path,
    log_path: Path,
    service_id: str,
    process_id: str,
    goal_audit_job_id: str,
    username: str,
    session_id: str,
    audit: dict[str, Any],
    history_sink: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    append_jsonl(
        session_goal_manager_reviews_path(runtime_root, username=username, session_id=session_id),
        {
            "ts": utc_ts(),
            "goal_audit_job_id": goal_audit_job_id,
            "goal_audit_session_id": audit["goal_audit_session_id"],
            "goal_id": str(audit.get("goal_id") or ""),
            "goal_text": str(audit.get("goal_text") or ""),
            "progress_state": audit["progress_state"],
            "audit_state": audit["audit_state"],
            "goal_satisfied": audit["goal_satisfied"],
            "summary": audit["summary"],
            "continue_xml": audit["continue_xml"],
            "request_compact": audit["request_compact"],
            "request_compact_reason": audit["request_compact_reason"],
            "agent_directives": list(audit.get("agent_directives", []))
            if isinstance(audit.get("agent_directives"), list)
            else [],
        },
    )
    _persist_goal_manager_runtime_state(
        runtime_root=runtime_root,
        username=username,
        session_id=session_id,
        payload={
            "state": "idle",
            "goal_audit_job_id": goal_audit_job_id,
            "goal_audit_session_id": audit["goal_audit_session_id"],
            "progress_state": audit["progress_state"],
            "audit_state": audit["audit_state"],
            "goal_satisfied": audit["goal_satisfied"],
            "summary": audit["summary"],
        },
    )
    write_jsonl(
        log_path,
        {
            "type": "service.goal_audit_completed",
            "ts": utc_ts(),
            "service_id": service_id,
            "process_id": process_id,
            "goal_audit_job_id": goal_audit_job_id,
            "scope": {"username": username, "session_id": session_id},
            "goal_audit_session_id": audit["goal_audit_session_id"],
            "goal_id": str(audit.get("goal_id") or ""),
            "goal_text": str(audit.get("goal_text") or ""),
            "progress_state": audit["progress_state"],
            "audit_state": audit["audit_state"],
            "goal_satisfied": audit["goal_satisfied"],
            "summary": audit["summary"],
            "continue_xml": audit["continue_xml"],
            "request_compact": audit["request_compact"],
            "request_compact_reason": audit["request_compact_reason"],
            "agent_directives": list(audit.get("agent_directives", []))
            if isinstance(audit.get("agent_directives"), list)
            else [],
        },
    )
    history_entry = {
        "direction": "agent",
        "ts": utc_ts(),
        "from": service_id,
        "session_id": session_id,
        "event_type": "service.goal_audit_completed",
        "text": goal_audit_history_text(audit),
        "event": {
            "type": "service.goal_audit_completed",
            "goal_audit_job_id": goal_audit_job_id,
            "goal_audit_session_id": audit["goal_audit_session_id"],
            "goal_id": str(audit.get("goal_id") or ""),
            "goal_text": str(audit.get("goal_text") or ""),
            "progress_state": audit["progress_state"],
            "audit_state": audit["audit_state"],
            "goal_satisfied": audit["goal_satisfied"],
            "summary": audit["summary"],
            "continue_xml": audit["continue_xml"],
            "request_compact": audit["request_compact"],
            "request_compact_reason": audit["request_compact_reason"],
            "agent_directives": list(audit.get("agent_directives", []))
            if isinstance(audit.get("agent_directives"), list)
            else [],
        },
    }
    if history_sink is not None:
        history_sink(history_entry)
        return
    append_user_history(
        runtime_root,
        username=username,
        session_id=session_id,
        entry=history_entry,
        limit=GOAL_AUDIT_HISTORY_LIMIT,
    )


def persist_goal_manager_compact_event(
    *,
    runtime_root: Path,
    log_path: Path,
    service_id: str,
    process_id: str,
    goal_audit_job_id: str,
    username: str,
    session_id: str,
    event: dict[str, Any],
    history_sink: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    _persist_goal_manager_runtime_state(
        runtime_root=runtime_root,
        username=username,
        session_id=session_id,
        payload={
            "state": "idle" if str(event.get("type") or "").endswith("_checked") else "failed",
            "compact_event": dict(event),
        },
    )
    write_jsonl(
        log_path,
        {
            "type": event["type"],
            "ts": utc_ts(),
            "service_id": service_id,
            "process_id": process_id,
            "goal_audit_job_id": goal_audit_job_id,
            "scope": {"username": username, "session_id": session_id},
            "event": event,
        },
    )
    persist_session_context_status(runtime_root, username=username, session_id=session_id, event=event, service_id=service_id)
    history_entry = make_history_event_entry(event, service_id=service_id)
    if history_sink is not None:
        history_sink(history_entry)
        return
    append_user_history(
        runtime_root,
        username=username,
        session_id=session_id,
        entry=history_entry,
        limit=GOAL_AUDIT_HISTORY_LIMIT,
    )


def persist_goal_manager_compact_started(
    *,
    runtime_root: Path,
    log_path: Path,
    service_id: str,
    process_id: str,
    goal_audit_job_id: str,
    username: str,
    session_id: str,
    reason: str,
    history_sink: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    event = {
        "type": "service.goal_manager_compact_started",
        "reason": reason,
        "session_id": session_id,
    }
    _persist_goal_manager_runtime_state(
        runtime_root=runtime_root,
        username=username,
        session_id=session_id,
        payload={
            "state": "running",
            "compact_reason": reason,
        },
    )
    write_jsonl(
        log_path,
        {
            "type": event["type"],
            "ts": utc_ts(),
            "service_id": service_id,
            "process_id": process_id,
            "goal_audit_job_id": goal_audit_job_id,
            "scope": {"username": username, "session_id": session_id},
            "event": event,
        },
    )
    history_entry = {
        "direction": "event",
        "ts": utc_ts(),
        "service_id": service_id,
        "event_type": event["type"],
        "text": f"GoalManager compact running: {reason or 'no reason provided'}",
        "event": event,
    }
    if history_sink is not None:
        history_sink(history_entry)
        return
    append_user_history(
        runtime_root,
        username=username,
        session_id=session_id,
        entry=history_entry,
        limit=GOAL_AUDIT_HISTORY_LIMIT,
    )


def handle_goal_manager_compact_request(
    *,
    runtime_root: Path,
    repo_root: Path,
    log_path: Path,
    service_id: str,
    process_id: str,
    goal_audit_job_id: str,
    username: str,
    session_id: str,
    audit: dict[str, Any],
    history_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any] | None:
    if not bool(audit.get("request_compact")):
        return None
    session_settings = get_session_settings(runtime_root, username=username, session_id=session_id) or {}
    compact_reason = str(audit.get("request_compact_reason", "") or "").strip()
    if bool(session_settings.get("goal_auto_compact_enabled", True)):
        persist_goal_manager_compact_started(
            runtime_root=runtime_root,
            log_path=log_path,
            service_id=service_id,
            process_id=process_id,
            goal_audit_job_id=goal_audit_job_id,
            username=username,
            session_id=session_id,
            reason=compact_reason,
            history_sink=history_sink,
        )
        try:
            service_kind = str(get_service_record(runtime_root, service_id).get("kind", "codex"))
        except (KeyError, FileNotFoundError):
            service_kind = "codex"
        # Import through cli_service_adapter so tests can patch the entrypoint that production uses.
        from runtime import cli_service_adapter as compact_entrypoints

        if service_kind == "claude":
            event, _returncode = compact_entrypoints.goal_manager_compact_claude_session(
                repo_root=repo_root,
                runtime_root=runtime_root,
                service_id=service_id,
                username=username,
                session_id=session_id,
            )
        else:
            event, _returncode = compact_entrypoints.goal_manager_compact_codex_session(
                repo_root=repo_root,
                runtime_root=runtime_root,
                service_id=service_id,
                username=username,
                session_id=session_id,
            )
        event["reason"] = compact_reason
    else:
        event = {
            "type": "service.goal_manager_compact_checked",
            "session_id": session_id,
            "compaction": "suppressed_by_session_setting",
            "left_percent": "?",
            "reason": compact_reason or "requested but suppressed by session setting",
        }
    persist_goal_manager_compact_event(
        runtime_root=runtime_root,
        log_path=log_path,
        service_id=service_id,
        process_id=process_id,
        goal_audit_job_id=goal_audit_job_id,
        username=username,
        session_id=session_id,
        event=event,
        history_sink=history_sink,
    )
    return event
