"""Batch processing over a store (T48/T49): one study + one run per input file,
sequential, failures isolated per file, report rewritten atomically after each
file so an interrupted batch can be resumed without redoing finished inputs.

The batch runs in-process through `Worker.process` (same code path as the API
worker) and must hold the store process lock: `serve` and `batch` never run
jobs on the same store at the same time (`max_active_jobs` = 1).
"""

import hashlib
import json
from pathlib import Path

from ecg_photo.contracts import sha256_file
from ecg_photo.ingest import IngestRejected
from ecg_photo.store import Job, RunNotFound, Store, StudyPatch, _atomic_json, _now
from ecg_photo.worker import EngineFactory, Worker

SCHEMA = "ecg-photo-batch/1"

# Outcomes that are final for identical bytes + options: skipped on --resume.
# failed / error / processing (crash mid-file) are retried.
FINAL_STATUSES = ("published", "completed_unpublished", "rejected", "cancelled")


class BatchError(RuntimeError):
    pass


def options_hash(options: StudyPatch) -> str:
    return hashlib.sha256(
        json.dumps(options.model_dump(mode="json", exclude_unset=True), sort_keys=True).encode()
    ).hexdigest()


def list_inputs(input_dir: Path) -> list[Path]:
    """Regular, non-hidden files directly in `input_dir`, sorted by name.
    Type admission is left to `ingest.admit` (unsupported → `rejected`)."""
    return sorted(
        p for p in Path(input_dir).iterdir() if p.is_file() and not p.name.startswith(".")
    )


def _summary(entries: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in entries:
        out[e["status"]] = out.get(e["status"], 0) + 1
    return out


def _pending_run(store: Store, study_id: str, prev: dict | None) -> Job | None:
    """The previous attempt's run if it is still queued and current (it was
    created but the process stopped before running it): run it, don't add one."""
    if prev is None or not prev.get("run_id"):
        return None
    state = store.get(study_id)
    try:
        job = store.get_job(study_id, prev["run_id"])
    except RunNotFound:
        return None
    if (
        state is not None
        and job.status == "queued"
        and not job.cancel_requested
        and state.selected_run_id == job.run_id
        and state.active_revision == job.input_revision
        and state.config_hash == job.config_hash
    ):
        return job
    return None


def run_batch(
    store: Store,
    engines: dict[str, EngineFactory],
    inputs: list[Path],
    options: StudyPatch,
    report_path: Path,
    *,
    author: str,
    reason: str,
    resume: bool = False,
) -> dict:
    if options.engine is None:
        raise BatchError("options.engine is required for a batch")
    if options.engine not in engines:
        raise BatchError(f"engine {options.engine!r} not configured on this worker")
    if options.lead_labels:
        raise BatchError("lead_labels are per-study corrections, not batch options")
    names = [p.name for p in inputs]
    if len(set(names)) != len(names):
        raise BatchError("duplicate input file names")

    report_path = Path(report_path)
    opt_hash = options_hash(options)
    previous: dict[str, dict] = {}
    if report_path.exists():
        if not resume:
            raise BatchError(f"{report_path} exists; pass --resume to continue it")
        old = json.loads(report_path.read_text(encoding="utf-8"))
        if old.get("schema") != SCHEMA:
            raise BatchError(f"{report_path} is not a {SCHEMA} report")
        if old.get("options_hash") != opt_hash:
            raise BatchError("options differ from the report being resumed")
        previous = {e["input"]: e for e in old.get("entries", [])}
        started_at = old.get("started_at", _now())
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = _now()

    worker = Worker(store, engines)  # not started: jobs run synchronously
    entries: list[dict] = []
    report: dict = {
        "schema": SCHEMA,
        "options": options.model_dump(mode="json", exclude_unset=True),
        "options_hash": opt_hash,
        "author": author,
        "reason": reason,
        "started_at": started_at,
        "updated_at": started_at,
        "entries": entries,
        "summary": {},
    }

    def _flush() -> None:
        report["updated_at"] = _now()
        report["summary"] = _summary(entries)
        _atomic_json(report_path, report)

    # keep entries of inputs no longer present, so the report stays complete
    present = set(names)
    entries.extend(e for n, e in previous.items() if n not in present)

    for path in inputs:
        digest = sha256_file(path)
        prev = previous.get(path.name)
        if prev is not None and prev.get("sha256") == digest and prev["status"] in FINAL_STATUSES:
            entries.append(prev)
            continue
        entry: dict = {
            "input": path.name,
            "sha256": digest,
            "study_id": None,
            "run_id": None,
            "status": "processing",
            "reason_code": None,
            "error": None,
            "attempts": 1,
        }
        if prev is not None and prev.get("sha256") == digest:
            entry["attempts"] = int(prev.get("attempts", 1)) + 1
            # retry on the study created by the previous attempt, if still there
            if prev.get("study_id") and store.get(prev["study_id"]) is not None:
                entry["study_id"] = prev["study_id"]
        entries.append(entry)
        _flush()
        try:
            if entry["study_id"] is None:
                state = store.create_study(
                    path,
                    path.name,
                    options=options,
                    author=author,
                    reason=reason,
                    method="initial options via batch",
                )
                entry["study_id"] = state.study_id
                _flush()
            else:
                found = store.get(entry["study_id"])
                assert found is not None
                state = found
            job = _pending_run(store, state.study_id, prev) or store.create_run(
                state.study_id, state.active_revision, state.config_hash
            )
            entry["run_id"] = job.run_id
            _flush()
            worker.process(state.study_id, job.run_id)
            job = store.get_job(state.study_id, job.run_id)
            published = store.get(state.study_id)
            if published is not None and published.published_run_id == job.run_id:
                entry["status"] = "published"
            else:
                entry["status"] = job.status
                entry["error"] = job.error
        except IngestRejected as e:
            entry["status"] = "rejected"
            entry["reason_code"] = str(e.reason_code)
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the batch
            entry["status"] = "error"
            entry["error"] = f"{type(e).__name__}: {e}"[:500]
        _flush()
    _flush()
    return report
