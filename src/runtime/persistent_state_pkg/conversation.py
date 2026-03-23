from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from wire.protocol import utc_ts

from ._core import (
    DEFAULT_AUTO_COMPACT_THRESHOLD_LEFT_PERCENT,
    DEFAULT_AUTO_RESUME_INTERVAL_SECONDS,
    DEFAULT_SESSION_GROUP,
    SESSION_GROUP_DEFAULT_PERMISSIONS,
    _active_goal_revision_unlocked,
    _apply_active_goal_snapshot_unlocked,
    _auth_sessions,
    _conversation_sessions,
    _ensure_default_session_unlocked,
    _ensure_session_defaults_unlocked,
    _ensure_session_exists_unlocked,
    _load_state_unlocked,
    write_state,
    digest_token,
    ensure_session_storage_unlocked,
    normalize_auto_compact_threshold_left_percent,
    normalize_username,
    read_json_file,
    read_jsonl,
    sessions_dir,
    session_dag_children_path,
    session_dag_parents_path,
    session_dir,
    session_metadata_path,
    session_goal_manager_state_path,
    session_timeline_path,
    session_user_dir,
    state_lock,
    state_read_lock,
    write_json_file,
)


def _parse_utc_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
        return datetime.fromisoformat(text).astimezone(UTC)
    except ValueError:
        return None


def _utc_ts_after_seconds(seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=max(0, int(seconds)))).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def session_group_permissions(session: dict[str, Any] | None) -> dict[str, bool]:
    record = dict(session or {})
    _ensure_session_defaults_unlocked(record)
    permissions = record.get("session_permissions")
    if isinstance(permissions, dict):
        return {str(key): bool(value) for key, value in permissions.items()}
    group = str(record.get("session_group") or DEFAULT_SESSION_GROUP).strip().lower()
    return {
        str(key): bool(value)
        for key, value in SESSION_GROUP_DEFAULT_PERMISSIONS.get(
            group,
            SESSION_GROUP_DEFAULT_PERMISSIONS[DEFAULT_SESSION_GROUP],
        ).items()
    }


def session_operation_allowed(session: dict[str, Any] | None, operation: str) -> bool:
    operation_name = str(operation or "").strip()
    if not operation_name:
        return False
    permissions = session_group_permissions(session)
    return bool(permissions.get(operation_name, False))


def _list_session_records(runtime_root: Path, *, username: str) -> list[dict[str, Any]]:
    user_dir = session_user_dir(runtime_root, username=username)
    if not user_dir.exists():
        return []
    sessions: list[dict[str, Any]] = []
    for talk_dir in sorted(path for path in user_dir.iterdir() if path.is_dir()):
        stored = read_json_file(session_metadata_path(runtime_root, username=username, session_id=talk_dir.name))
        if not isinstance(stored, dict):
            continue
        _ensure_session_defaults_unlocked(stored)
        ensure_session_storage_unlocked(runtime_root, username=username, session=stored)
        sessions.append(dict(stored))
    return sessions


def list_sessions(runtime_root: Path, *, username: str) -> list[dict[str, Any]]:
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        _ensure_default_session_unlocked(state, normalized)
        return _list_session_records(runtime_root, username=normalized)


def list_sessions_with_histories(
    runtime_root: Path, *, username: str
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        _ensure_default_session_unlocked(state, normalized)
        sessions = _list_session_records(runtime_root, username=normalized)
        histories: dict[str, list[dict[str, Any]]] = {}
        for session in sessions:
            session_id = str(session.get("session_id") or "")
            timeline_path = session_timeline_path(runtime_root, username=normalized, session_id=session_id)
            histories[session_id] = read_jsonl(timeline_path)
    return sessions, histories


def list_all_sessions_with_users(runtime_root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    sessions_root = sessions_dir(runtime_root)
    with state_read_lock(runtime_root):
        if not sessions_root.exists():
            return result
        for user_dir in sorted(path for path in sessions_root.iterdir() if path.is_dir()):
            username = normalize_username(user_dir.name)
            for session in _list_session_records(runtime_root, username=username):
                entry = dict(session)
                entry["username"] = username
                result.append(entry)
    return result


def list_sessions_bound_to_service(runtime_root: Path, *, service_id: str) -> list[dict[str, Any]]:
    bound: list[dict[str, Any]] = []
    with state_lock(runtime_root):
        sessions_root = sessions_dir(runtime_root)
        if not sessions_root.exists():
            return bound
        for user_dir in sorted(path for path in sessions_root.iterdir() if path.is_dir()):
            username = normalize_username(user_dir.name)
            for session in _list_session_records(runtime_root, username=username):
                if str(session.get("service_id") or "") != service_id:
                    continue
                bound.append(
                    {
                        "username": username,
                        "session_id": str(session.get("session_id") or ""),
                        "service_id": service_id,
                    }
                )
    return bound


def create_conversation_session(
    runtime_root: Path,
    *,
    username: str,
    label: str | None = None,
    session_group: str | None = None,
    session_permissions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        sessions = _conversation_sessions(state).setdefault(normalized, [])
        normalized_group = str(session_group or DEFAULT_SESSION_GROUP).strip().lower() or DEFAULT_SESSION_GROUP
        if normalized_group not in SESSION_GROUP_DEFAULT_PERMISSIONS:
            normalized_group = DEFAULT_SESSION_GROUP
        defaults = SESSION_GROUP_DEFAULT_PERMISSIONS.get(
            normalized_group,
            SESSION_GROUP_DEFAULT_PERMISSIONS[DEFAULT_SESSION_GROUP],
        )
        normalized_permissions: dict[str, bool] = {}
        if isinstance(session_permissions, dict):
            for operation_name, default_value in defaults.items():
                if operation_name in session_permissions:
                    normalized_permissions[operation_name] = bool(session_permissions.get(operation_name))
                else:
                    normalized_permissions[operation_name] = bool(default_value)
        else:
            normalized_permissions = {str(key): bool(value) for key, value in defaults.items()}
        session = {
            "session_id": secrets.token_hex(8),
            "label": (label or "").strip() or f"Session {len(sessions) + 1}",
            "session_group": normalized_group,
            "session_permissions": normalized_permissions,
            "auto_compact_threshold_left_percent": DEFAULT_AUTO_COMPACT_THRESHOLD_LEFT_PERCENT,
            "created_at": utc_ts(),
            "updated_at": utc_ts(),
        }
        sessions.append(session)
        ensure_session_storage_unlocked(runtime_root, username=normalized, session=session)
        write_state(runtime_root, state)
        return dict(session)


def get_session_service(runtime_root: Path, *, username: str, session_id: str) -> str | None:
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        session = read_json_file(session_metadata_path(runtime_root, username=normalized, session_id=session_id))
        if isinstance(session, dict):
            service_id = session.get("service_id")
            return str(service_id) if isinstance(service_id, str) and service_id else None
        return None


def lease_session_service(
    runtime_root: Path,
    *,
    username: str,
    session_id: str,
    pool_service_ids: list[str],
) -> str | None:
    normalized = normalize_username(username)
    pool = [service_id for service_id in pool_service_ids if isinstance(service_id, str) and service_id]
    if not pool:
        return None
    with state_lock(runtime_root):
        session_path = session_metadata_path(runtime_root, username=normalized, session_id=session_id)
        target_session = read_json_file(session_path)
        if not isinstance(target_session, dict):
            return None
        _ensure_session_defaults_unlocked(target_session)
        existing_service_id = target_session.get("service_id")
        if isinstance(existing_service_id, str) and existing_service_id in pool:
            return existing_service_id
        leased: set[str] = set()
        sessions_root = runtime_root.parent / ".aize-state" / "sessions"
        if sessions_root.exists():
            for user_dir in sorted(path for path in sessions_root.iterdir() if path.is_dir()):
                for talk_dir in sorted(path for path in user_dir.iterdir() if path.is_dir()):
                    talk = read_json_file(session_metadata_path(runtime_root, username=user_dir.name, session_id=talk_dir.name))
                    if not isinstance(talk, dict):
                        continue
                    service_id = talk.get("service_id")
                    if isinstance(service_id, str) and service_id in pool:
                        leased.add(service_id)
        available = next((service_id for service_id in pool if service_id not in leased), None)
        if available is None:
            return None
        target_session["service_id"] = available
        target_session["updated_at"] = utc_ts()
        ensure_session_storage_unlocked(runtime_root, username=normalized, session=target_session)
        return available


def release_session_service(runtime_root: Path, *, username: str, session_id: str) -> str | None:
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        session_path = session_metadata_path(runtime_root, username=normalized, session_id=session_id)
        session = read_json_file(session_path)
        if not isinstance(session, dict):
            return None
        _ensure_session_defaults_unlocked(session)
        service_id = session.get("service_id")
        if not isinstance(service_id, str) or not service_id:
            return None
        session.pop("service_id", None)
        session["updated_at"] = utc_ts()
        ensure_session_storage_unlocked(runtime_root, username=normalized, session=session)
        return service_id


def release_nonrunnable_session_services(runtime_root: Path) -> list[dict[str, str]]:
    from kernel.lifecycle import load_lifecycle_state
    from kernel.registry import load_registry

    services = load_registry(runtime_root).get("services", {})
    processes = load_lifecycle_state(runtime_root).get("processes", {})
    terminal_statuses = {"stopped", "failed", "crashed", "dead", "exited"}

    def release_reason_for_service(service_id: str) -> str | None:
        service = services.get(service_id)
        if not isinstance(service, dict):
            return "service_missing"
        service_status = str(service.get("status") or "").strip().lower()
        if service_status in terminal_statuses:
            return f"service_status:{service_status}"
        current_process_id = service.get("current_process_id")
        if not isinstance(current_process_id, str) or not current_process_id:
            return None
        process = processes.get(current_process_id)
        if not isinstance(process, dict):
            return "process_missing"
        process_status = str(process.get("status") or "").strip().lower()
        if process_status in terminal_statuses:
            return f"process_status:{process_status}"
        return None

    released: list[dict[str, str]] = []
    with state_lock(runtime_root):
        sessions_root = runtime_root.parent / ".aize-state" / "sessions"
        if not sessions_root.exists():
            return released
        for user_dir in sorted(path for path in sessions_root.iterdir() if path.is_dir()):
            username = normalize_username(user_dir.name)
            for talk_dir in sorted(path for path in user_dir.iterdir() if path.is_dir()):
                session_path = session_metadata_path(runtime_root, username=username, session_id=talk_dir.name)
                talk = read_json_file(session_path)
                if not isinstance(talk, dict):
                    continue
                service_id = talk.get("service_id")
                if not isinstance(service_id, str) or not service_id:
                    continue
                _ensure_session_defaults_unlocked(talk)
                goal_active = bool(talk.get("goal_active", False))
                goal_completed = bool(talk.get("goal_completed", False))
                goal_progress_state = str(
                    talk.get("goal_progress_state", "complete" if goal_completed else "in_progress")
                ).strip().lower()
                release_reason = release_reason_for_service(service_id)
                if goal_completed or not goal_active or goal_progress_state == "complete" or release_reason:
                    released.append(
                        {
                            "username": username,
                            "session_id": str(talk.get("session_id", "")),
                            "service_id": service_id,
                            "reason": release_reason or "goal_inactive",
                        }
                    )
                    talk.pop("service_id", None)
                    talk["updated_at"] = utc_ts()
                    write_json_file(session_path, talk)
    return released


def get_session_settings(runtime_root: Path, *, username: str, session_id: str) -> dict[str, Any] | None:
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        stored = read_json_file(session_metadata_path(runtime_root, username=normalized, session_id=session_id))
        if isinstance(stored, dict):
            _ensure_session_defaults_unlocked(stored)
            ensure_session_storage_unlocked(runtime_root, username=normalized, session=stored)
            return dict(stored)
        return None


def update_session_auto_compact_threshold(
    runtime_root: Path,
    *,
    username: str,
    session_id: str,
    threshold_left_percent: Any,
) -> dict[str, Any] | None:
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        if not _ensure_session_exists_unlocked(state, normalized, session_id):
            return None
        normalized_threshold = normalize_auto_compact_threshold_left_percent(threshold_left_percent)
        for talk in _conversation_sessions(state).get(normalized, []):
            if isinstance(talk, dict) and str(talk.get("session_id")) == session_id:
                _ensure_session_defaults_unlocked(talk)
                talk["auto_compact_threshold_left_percent"] = normalized_threshold
                talk["updated_at"] = utc_ts()
                ensure_session_storage_unlocked(runtime_root, username=normalized, session=talk)
                return dict(talk)
        return None


def update_session_goal(
    runtime_root: Path,
    *,
    username: str,
    session_id: str,
    goal_text: Any,
) -> dict[str, Any] | None:
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        if not _ensure_session_exists_unlocked(state, normalized, session_id):
            return None
        normalized_goal = str(goal_text or "").strip()
        for talk in _conversation_sessions(state).get(normalized, []):
            if isinstance(talk, dict) and str(talk.get("session_id")) == session_id:
                _ensure_session_defaults_unlocked(talk)
                previous_goal = _active_goal_revision_unlocked(talk)
                revision_ts = utc_ts()
                new_revision = {
                    "goal_id": secrets.token_hex(8),
                    "previous_goal_id": (
                        str(previous_goal.get("goal_id") or "").strip() or None
                    ) if isinstance(previous_goal, dict) else None,
                    "goal_text": normalized_goal,
                    "goal_active": bool(normalized_goal),
                    "goal_completed": False,
                    "goal_progress_state": "in_progress",
                    "created_at": revision_ts,
                    "updated_at": revision_ts,
                }
                goal_history = talk.get("goal_history")
                if not isinstance(goal_history, list):
                    goal_history = []
                goal_history.append(new_revision)
                talk["goal_history"] = goal_history
                talk["active_goal_id"] = str(new_revision["goal_id"])
                _apply_active_goal_snapshot_unlocked(talk)
                talk["updated_at"] = utc_ts()
                ensure_session_storage_unlocked(runtime_root, username=normalized, session=talk)
                return dict(talk)
        return None


def update_session_goal_flags(
    runtime_root: Path,
    *,
    username: str,
    session_id: str,
    goal_id: Any | None = None,
    goal_active: Any | None = None,
    goal_completed: Any | None = None,
    goal_progress_state: Any | None = None,
    goal_audit_state: Any | None = None,
    goal_reset_completed_on_prompt: Any | None = None,
    goal_auto_compact_enabled: Any | None = None,
    agent_welcome_enabled: Any | None = None,
    preferred_provider: Any | None = None,
    auto_resume_enabled: Any | None = None,
    auto_resume_interval_seconds: Any | None = None,
) -> dict[str, Any] | None:
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        if not _ensure_session_exists_unlocked(state, normalized, session_id):
            return None
        for talk in _conversation_sessions(state).get(normalized, []):
            if isinstance(talk, dict) and str(talk.get("session_id")) == session_id:
                _ensure_session_defaults_unlocked(talk)
                target_goal_id = str(goal_id or talk.get("active_goal_id") or "").strip()
                target_revision: dict[str, Any] | None = None
                goal_history = talk.get("goal_history")
                if isinstance(goal_history, list):
                    for revision in goal_history:
                        if isinstance(revision, dict) and str(revision.get("goal_id") or "").strip() == target_goal_id:
                            target_revision = revision
                            break
                if target_revision is None:
                    target_revision = _active_goal_revision_unlocked(talk)

                goal_status_updates = any(
                    update is not None
                    for update in (
                        goal_active,
                        goal_completed,
                        goal_progress_state,
                        goal_audit_state,
                    )
                )
                if target_revision is None and goal_status_updates:
                    if not isinstance(goal_history, list):
                        goal_history = []
                    revision_ts = utc_ts()
                    goal_text = str(talk.get("goal_text") or "").strip()
                    target_revision = {
                        "goal_id": secrets.token_hex(8),
                        "previous_goal_id": (
                            str(talk.get("goal_id") or talk.get("active_goal_id") or "").strip() or None
                        ),
                        "goal_text": goal_text,
                        "goal_active": bool(goal_active) if goal_active is not None else bool(goal_text),
                        "goal_completed": bool(goal_completed) if goal_completed is not None else bool(talk.get("goal_completed", False)),
                        "goal_progress_state": "in_progress",
                        "created_at": revision_ts,
                        "updated_at": revision_ts,
                    }
                    if goal_progress_state is not None:
                        normalized_progress_state = str(goal_progress_state).strip().lower()
                        target_revision["goal_progress_state"] = (
                            normalized_progress_state
                            if normalized_progress_state in {"complete", "in_progress"}
                            else "in_progress"
                        )
                        target_revision["goal_completed"] = (
                            target_revision["goal_progress_state"] == "complete"
                        )
                    goal_history.append(target_revision)
                    talk["goal_history"] = goal_history
                    talk["active_goal_id"] = str(target_revision["goal_id"])
                if target_revision is not None:
                    if goal_active is not None:
                        target_revision["goal_active"] = bool(goal_active)
                        if bool(goal_active):
                            talk["active_goal_id"] = str(target_revision.get("goal_id") or "")
                    if goal_completed is not None:
                        target_revision["goal_completed"] = bool(goal_completed)
                        target_revision["goal_progress_state"] = (
                            "complete" if bool(goal_completed) else "in_progress"
                        )
                    if goal_progress_state is not None:
                        progress_state = str(goal_progress_state).strip().lower()
                        target_revision["goal_progress_state"] = (
                            progress_state if progress_state in {"complete", "in_progress"} else "in_progress"
                        )
                        target_revision["goal_completed"] = (
                            target_revision["goal_progress_state"] == "complete"
                        )
                    target_revision["updated_at"] = utc_ts()
                # goal_audit_state is now agent-side; the parameter is kept for call-site compatibility
                # but no longer written to the session record
                if goal_reset_completed_on_prompt is not None:
                    talk["goal_reset_completed_on_prompt"] = bool(goal_reset_completed_on_prompt)
                if goal_auto_compact_enabled is not None:
                    talk["goal_auto_compact_enabled"] = bool(goal_auto_compact_enabled)
                if agent_welcome_enabled is not None:
                    talk["agent_welcome_enabled"] = bool(agent_welcome_enabled)
                if preferred_provider is not None:
                    provider = str(preferred_provider).strip().lower()
                    talk["preferred_provider"] = provider if provider in {"codex", "claude"} else "codex"
                if auto_resume_enabled is not None:
                    talk["auto_resume_enabled"] = bool(auto_resume_enabled)
                    if not bool(auto_resume_enabled):
                        talk["auto_resume_next_at"] = ""
                        talk["auto_resume_reason"] = ""
                if auto_resume_interval_seconds is not None:
                    try:
                        talk["auto_resume_interval_seconds"] = max(300, int(auto_resume_interval_seconds))
                    except (TypeError, ValueError):
                        talk["auto_resume_interval_seconds"] = DEFAULT_AUTO_RESUME_INTERVAL_SECONDS
                _apply_active_goal_snapshot_unlocked(talk)
                if goal_completed is not None:
                    if bool(goal_completed) and bool(talk.get("goal_active", False)) and bool(talk.get("auto_resume_enabled", False)):
                        talk["auto_resume_next_at"] = _utc_ts_after_seconds(
                            int(talk.get("auto_resume_interval_seconds", DEFAULT_AUTO_RESUME_INTERVAL_SECONDS))
                        )
                        talk["auto_resume_reason"] = "goal_completed_interval"
                        talk["auto_resume_last_scheduled_at"] = utc_ts()
                    elif not bool(goal_completed):
                        talk["auto_resume_next_at"] = ""
                        talk["auto_resume_reason"] = ""
                        talk["auto_resume_last_error"] = ""
                        talk["auto_resume_last_started_at"] = utc_ts()
                talk["updated_at"] = utc_ts()
                ensure_session_storage_unlocked(runtime_root, username=normalized, session=talk)
                return dict(talk)
        return None


def schedule_session_auto_resume(
    runtime_root: Path,
    *,
    username: str,
    session_id: str,
    reason: str,
    error_text: str = "",
    retry_after_seconds: Any | None = None,
    mark_completed: bool = True,
) -> dict[str, Any] | None:
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        if not _ensure_session_exists_unlocked(state, normalized, session_id):
            return None
        for talk in _conversation_sessions(state).get(normalized, []):
            if not isinstance(talk, dict) or str(talk.get("session_id")) != session_id:
                continue
            _ensure_session_defaults_unlocked(talk)
            if not bool(talk.get("auto_resume_enabled", False)):
                talk["auto_resume_next_at"] = ""
                talk["auto_resume_reason"] = ""
                talk["auto_resume_last_error"] = str(error_text or "")
                talk["updated_at"] = utc_ts()
                ensure_session_storage_unlocked(runtime_root, username=normalized, session=talk)
                return dict(talk)
            try:
                delay_seconds = int(
                    retry_after_seconds
                    if retry_after_seconds is not None
                    else talk.get("auto_resume_interval_seconds", DEFAULT_AUTO_RESUME_INTERVAL_SECONDS)
                )
            except (TypeError, ValueError):
                delay_seconds = DEFAULT_AUTO_RESUME_INTERVAL_SECONDS
            talk["auto_resume_next_at"] = _utc_ts_after_seconds(delay_seconds)
            talk["auto_resume_reason"] = str(reason or "").strip()
            talk["auto_resume_last_error"] = str(error_text or "").strip()
            talk["auto_resume_last_scheduled_at"] = utc_ts()
            if mark_completed and bool(talk.get("goal_active", False)):
                talk["goal_completed"] = True
                talk["goal_progress_state"] = "complete"
            talk["updated_at"] = utc_ts()
            ensure_session_storage_unlocked(runtime_root, username=normalized, session=talk)
            return dict(talk)
        return None


def consume_session_due_auto_resume(
    runtime_root: Path,
    *,
    username: str,
    session_id: str,
) -> dict[str, Any] | None:
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        if not _ensure_session_exists_unlocked(state, normalized, session_id):
            return None
        now = datetime.now(UTC)
        for talk in _conversation_sessions(state).get(normalized, []):
            if not isinstance(talk, dict) or str(talk.get("session_id")) != session_id:
                continue
            _ensure_session_defaults_unlocked(talk)
            if not bool(talk.get("auto_resume_enabled", False)):
                return None
            if not bool(talk.get("goal_active", False)) or not bool(talk.get("goal_completed", False)):
                return None
            if not session_operation_allowed(talk, "auto_resume"):
                return None
            due_at = _parse_utc_ts(talk.get("auto_resume_next_at"))
            if due_at is None or due_at > now:
                return None
            talk["goal_completed"] = False
            talk["goal_progress_state"] = "in_progress"
            talk["auto_resume_last_started_at"] = utc_ts()
            talk["auto_resume_next_at"] = ""
            talk["auto_resume_reason"] = ""
            talk["updated_at"] = utc_ts()
            ensure_session_storage_unlocked(runtime_root, username=normalized, session=talk)
            return dict(talk)
        return None


def update_session_context_status(
    runtime_root: Path,
    *,
    username: str,
    session_id: str,
    context_status: dict[str, Any] | None,
) -> dict[str, Any] | None:
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        if not _ensure_session_exists_unlocked(state, normalized, session_id):
            return None
        for talk in _conversation_sessions(state).get(normalized, []):
            if isinstance(talk, dict) and str(talk.get("session_id")) == session_id:
                _ensure_session_defaults_unlocked(talk)
                talk["last_context_status"] = dict(context_status) if isinstance(context_status, dict) else None
                talk["last_context_status_updated_at"] = utc_ts()
                talk["updated_at"] = utc_ts()
                ensure_session_storage_unlocked(runtime_root, username=normalized, session=talk)
                return dict(talk)
        return None


def record_session_agent_contact(
    runtime_root: Path,
    *,
    username: str,
    session_id: str,
    service_id: str,
    agent_id: str | None = None,
    provider: str | None = None,
    turn_completed_at: str | None = None,
) -> dict[str, Any] | None:
    normalized = normalize_username(username)
    normalized_service_id = str(service_id or "").strip()
    normalized_agent_id = str(agent_id or "").strip()
    if not normalized_service_id:
        return None
    normalized_provider = str(provider or "").strip().lower()
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        if not _ensure_session_exists_unlocked(state, normalized, session_id):
            return None
        for talk in _conversation_sessions(state).get(normalized, []):
            if isinstance(talk, dict) and str(talk.get("session_id")) == session_id:
                _ensure_session_defaults_unlocked(talk)
                welcomed_agents = talk.get("welcomed_agents")
                if not isinstance(welcomed_agents, list):
                    welcomed_agents = []
                existing: dict[str, Any] | None = None
                for item in welcomed_agents:
                    if isinstance(item, dict) and str(item.get("service_id") or "").strip() == normalized_service_id:
                        existing = item
                        break
                now = utc_ts()
                if existing is None:
                    if not normalized_agent_id:
                        normalized_agent_id = f"{normalized_service_id}@@{session_id}"
                    existing = {
                        "agent_id": normalized_agent_id,
                        "service_id": normalized_service_id,
                        "provider": normalized_provider,
                        "welcomed_at": now,
                        "last_turn_completed_at": "",
                        "updated_at": now,
                    }
                    welcomed_agents.append(existing)
                else:
                    existing["updated_at"] = now
                    if not str(existing.get("agent_id") or "").strip():
                        existing["agent_id"] = normalized_agent_id or f"{normalized_service_id}@@{session_id}"
                if normalized_provider:
                    existing["provider"] = normalized_provider
                if isinstance(turn_completed_at, str) and turn_completed_at.strip():
                    existing["last_turn_completed_at"] = turn_completed_at.strip()
                talk["welcomed_agents"] = welcomed_agents
                talk["updated_at"] = now
                ensure_session_storage_unlocked(runtime_root, username=normalized, session=talk)
                return dict(talk)
        return None


def resolve_session_agent_id(
    runtime_root: Path,
    *,
    username: str,
    session_id: str,
    service_id: str,
) -> str:
    normalized_service_id = str(service_id or "").strip()
    if not normalized_service_id:
        return ""
    welcomed_agents = list_session_agent_contacts(runtime_root, username=username, session_id=session_id)
    for item in welcomed_agents:
        if str(item.get("service_id") or "").strip() == normalized_service_id:
            agent_id = str(item.get("agent_id") or "").strip()
            if agent_id:
                return agent_id
    return f"{normalized_service_id}@@{session_id}"


def list_session_agent_contacts(
    runtime_root: Path,
    *,
    username: str,
    session_id: str,
) -> list[dict[str, Any]]:
    talk = get_session_settings(runtime_root, username=username, session_id=session_id)
    if not isinstance(talk, dict):
        return []
    welcomed_agents = talk.get("welcomed_agents")
    if not isinstance(welcomed_agents, list):
        return []
    return [dict(item) for item in welcomed_agents if isinstance(item, dict)]


def update_goal_manager_review_cursor(
    runtime_root: Path,
    *,
    username: str,
    session_id: str,
    last_turn_completed_at: str,
) -> dict[str, Any] | None:
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        if not _ensure_session_exists_unlocked(state, normalized, session_id):
            return None
        for talk in _conversation_sessions(state).get(normalized, []):
            if isinstance(talk, dict) and str(talk.get("session_id")) == session_id:
                _ensure_session_defaults_unlocked(talk)
                talk["goal_manager_last_reviewed_turn_completed_at"] = str(last_turn_completed_at or "").strip()
                talk["updated_at"] = utc_ts()
                ensure_session_storage_unlocked(runtime_root, username=normalized, session=talk)
                goal_manager_state_path = session_goal_manager_state_path(
                    runtime_root,
                    username=normalized,
                    session_id=session_id,
                )
                goal_manager_state = read_json_file(goal_manager_state_path) or {}
                goal_manager_state["last_reviewed_turn_completed_at"] = talk["goal_manager_last_reviewed_turn_completed_at"]
                goal_manager_state["updated_at"] = talk["updated_at"]
                write_json_file(goal_manager_state_path, goal_manager_state)
                return dict(talk)
        return None


def rename_session(runtime_root: Path, *, username: str, session_id: str, label: str) -> dict[str, Any] | None:
    normalized = normalize_username(username)
    normalized_label = str(label or "").strip()
    if not normalized_label:
        return None
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        if not _ensure_session_exists_unlocked(state, normalized, session_id):
            return None
        for talk in _conversation_sessions(state).get(normalized, []):
            if isinstance(talk, dict) and str(talk.get("session_id")) == session_id:
                _ensure_session_defaults_unlocked(talk)
                talk["label"] = normalized_label
                talk["updated_at"] = utc_ts()
                ensure_session_storage_unlocked(runtime_root, username=normalized, session=talk)
                return dict(talk)
        return None


def select_session(runtime_root: Path, *, token: str, session_id: str) -> dict[str, str] | None:
    token_hash = digest_token(token)
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        session = _auth_sessions(state).get(token_hash)
        if not session:
            return None
        username = str(session["username"])
        if not _ensure_session_exists_unlocked(state, username, session_id):
            return None
        session["active_session_id"] = session_id
        session["updated_at"] = utc_ts()
        write_state(runtime_root, state)
        return {"username": username, "session_id": session_id}


def _load_dag_ids(path: Path, key: str) -> list[str]:
    payload = read_json_file(path) or {}
    values = payload.get(key)
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _write_dag_ids(path: Path, key: str, values: list[str]) -> None:
    write_json_file(path, {key: values})


def _list_session_parents_unlocked(runtime_root: Path, *, username: str, session_id: str) -> list[str]:
    return _load_dag_ids(
        session_dag_parents_path(runtime_root, username=username, session_id=session_id),
        "parents",
    )


def _list_session_children_unlocked(runtime_root: Path, *, username: str, session_id: str) -> list[str]:
    return _load_dag_ids(
        session_dag_children_path(runtime_root, username=username, session_id=session_id),
        "children",
    )


def list_session_parents(runtime_root: Path, *, username: str, session_id: str) -> list[str]:
    normalized = normalize_username(username)
    with state_read_lock(runtime_root):
        return _list_session_parents_unlocked(runtime_root, username=normalized, session_id=session_id)


def list_session_children(runtime_root: Path, *, username: str, session_id: str) -> list[str]:
    normalized = normalize_username(username)
    with state_read_lock(runtime_root):
        return _list_session_children_unlocked(runtime_root, username=normalized, session_id=session_id)


def _session_descends_from(
    runtime_root: Path,
    *,
    username: str,
    session_id: str,
    ancestor_session_id: str,
) -> bool:
    if session_id == ancestor_session_id:
        return True
    seen: set[str] = set()
    stack = [session_id]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        parents = _list_session_parents_unlocked(runtime_root, username=username, session_id=current)
        if ancestor_session_id in parents:
            return True
        stack.extend(parents)
    return False


def add_session_child(
    runtime_root: Path,
    *,
    username: str,
    parent_session_id: str,
    child_session_id: str,
) -> dict[str, list[str]] | None:
    normalized = normalize_username(username)
    if not child_session_id or parent_session_id == child_session_id:
        return None
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        if not _ensure_session_exists_unlocked(state, normalized, parent_session_id):
            return None
        if not _ensure_session_exists_unlocked(state, normalized, child_session_id):
            return None
        if _session_descends_from(
            runtime_root,
            username=normalized,
            session_id=parent_session_id,
            ancestor_session_id=child_session_id,
        ):
            raise ValueError("session_dag_cycle")
        parent_children = _list_session_children_unlocked(
            runtime_root,
            username=normalized,
            session_id=parent_session_id,
        )
        if child_session_id not in parent_children:
            parent_children.append(child_session_id)
        _write_dag_ids(
            session_dag_children_path(runtime_root, username=normalized, session_id=parent_session_id),
            "children",
            parent_children,
        )
        child_parents = _list_session_parents_unlocked(
            runtime_root,
            username=normalized,
            session_id=child_session_id,
        )
        if parent_session_id not in child_parents:
            child_parents.append(parent_session_id)
        _write_dag_ids(
            session_dag_parents_path(runtime_root, username=normalized, session_id=child_session_id),
            "parents",
            child_parents,
        )
        parent_session = read_json_file(
            session_metadata_path(runtime_root, username=normalized, session_id=parent_session_id)
        ) or {}
        parent_session["waiting_on_children"] = True
        parent_session["updated_at"] = utc_ts()
        if isinstance(parent_session.get("goal_progress_state"), str) and parent_session.get("goal_progress_state") == "complete":
            parent_session["goal_progress_state"] = "in_progress"
            parent_session["goal_completed"] = False
        ensure_session_storage_unlocked(runtime_root, username=normalized, session=parent_session)
        child_session = read_json_file(
            session_metadata_path(runtime_root, username=normalized, session_id=child_session_id)
        ) or {}
        child_session["parent_session_id"] = parent_session_id
        child_session["updated_at"] = utc_ts()
        ensure_session_storage_unlocked(runtime_root, username=normalized, session=child_session)
        return {"parents": child_parents, "children": parent_children}


def create_child_conversation_session(
    runtime_root: Path,
    *,
    username: str,
    parent_session_id: str,
    label: str | None = None,
    goal_text: str | None = None,
    session_group: str | None = None,
    session_permissions: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    parent = get_session_settings(runtime_root, username=username, session_id=parent_session_id)
    if not isinstance(parent, dict):
        return None
    if not session_operation_allowed(parent, "create_child_session"):
        return None
    child = create_conversation_session(
        runtime_root,
        username=username,
        label=label or "Subgoal",
        session_group=session_group,
        session_permissions=session_permissions,
    )
    if goal_text:
        update_session_goal(runtime_root, username=username, session_id=str(child["session_id"]), goal_text=goal_text)
    add_session_child(
        runtime_root,
        username=username,
        parent_session_id=parent_session_id,
        child_session_id=str(child["session_id"]),
    )
    return get_session_settings(runtime_root, username=username, session_id=str(child["session_id"]))


def complete_session_child(
    runtime_root: Path,
    *,
    username: str,
    parent_session_id: str,
    child_session_id: str,
) -> dict[str, Any] | None:
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        if not _ensure_session_exists_unlocked(state, normalized, parent_session_id):
            return None
        if not _ensure_session_exists_unlocked(state, normalized, child_session_id):
            return None
        child_parents = _list_session_parents_unlocked(
            runtime_root,
            username=normalized,
            session_id=child_session_id,
        )
        if parent_session_id not in child_parents:
            return None
        parent_children = _list_session_children_unlocked(
            runtime_root,
            username=normalized,
            session_id=parent_session_id,
        )
        remaining_children: list[str] = []
        for sibling_session_id in parent_children:
            sibling_session = read_json_file(
                session_metadata_path(runtime_root, username=normalized, session_id=sibling_session_id)
            ) or {}
            sibling_progress_state = str(
                sibling_session.get(
                    "goal_progress_state",
                    "complete" if bool(sibling_session.get("goal_completed", False)) else "in_progress",
                )
            ).strip().lower()
            if sibling_session_id == child_session_id:
                sibling_session["child_completion_reported_at"] = utc_ts()
                sibling_session["updated_at"] = utc_ts()
                ensure_session_storage_unlocked(runtime_root, username=normalized, session=sibling_session)
            if sibling_progress_state != "complete":
                remaining_children.append(sibling_session_id)
        parent_session = read_json_file(
            session_metadata_path(runtime_root, username=normalized, session_id=parent_session_id)
        ) or {}
        parent_session["waiting_on_children"] = bool(remaining_children)
        parent_session["updated_at"] = utc_ts()
        ensure_session_storage_unlocked(runtime_root, username=normalized, session=parent_session)
        return {
            "parent_session_id": parent_session_id,
            "child_session_id": child_session_id,
            "waiting_on_children": bool(remaining_children),
            "remaining_children": remaining_children,
        }


def update_session_peer_joinable(
    runtime_root: Path,
    *,
    username: str,
    session_id: str,
    peer_joinable: bool,
) -> dict[str, Any] | None:
    """Set or clear the peer_joinable flag on a session.

    When ``peer_joinable`` is ``True`` the session is visible to remote AIze
    peers connecting via ``/ws`` and they may join it as external agents.
    """
    normalized = normalize_username(username)
    with state_lock(runtime_root):
        state = _load_state_unlocked(runtime_root)
        if not _ensure_session_exists_unlocked(state, normalized, session_id):
            return None
        for talk in _conversation_sessions(state).get(normalized, []):
            if isinstance(talk, dict) and str(talk.get("session_id")) == session_id:
                _ensure_session_defaults_unlocked(talk)
                talk["peer_joinable"] = bool(peer_joinable)
                talk["updated_at"] = utc_ts()
                ensure_session_storage_unlocked(runtime_root, username=normalized, session=talk)
                return dict(talk)
    return None


def list_peer_joinable_sessions(runtime_root: Path) -> list[dict[str, Any]]:
    """Return all sessions across all users that have ``peer_joinable=True``.

    Each entry includes a ``username`` key so remote peers know which user
    owns the session.
    """
    result: list[dict[str, Any]] = []
    sessions_root = sessions_dir(runtime_root)
    with state_read_lock(runtime_root):
        if not sessions_root.exists():
            return result
        for user_dir in sorted(path for path in sessions_root.iterdir() if path.is_dir()):
            username = normalize_username(user_dir.name)
            for session in _list_session_records(runtime_root, username=username):
                if not bool(session.get("peer_joinable")):
                    continue
                entry = {
                    "username": username,
                    "session_id": str(session.get("session_id") or ""),
                    "label": str(session.get("label") or session.get("session_id") or ""),
                    "goal_text": str(session.get("goal_text") or ""),
                    "peer_joinable": True,
                }
                result.append(entry)
    return result
