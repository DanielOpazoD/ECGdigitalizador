"""Signal quality control of a confirmed run (read-only report, no truth needed).

Per lead: coverage of the expected slot, flat/empty trace, missing lead, and
the rhythm strip duration against the layout. Across leads: the limb-lead
identities that hold for any real ECG and share a time column in the
3x4 + rhythm layout (Einthoven I + III = II; Goldberger aVR + aVL + aVF = 0).
The identity residuals are normalised rms errors, so they measure how far a
digitization is from a physically possible ECG.

Thresholds were set on the F6-b Kaggle bench (Ahus, engine axis). Einthoven
residual medians: originals 0.10, flat scans 0.14-0.35, phone photos
0.34-0.53. Against the truth, leads stay good up to a residual of 1.0 (median
r@lag 0.85-0.95 in 0.7-1.0) and degrade above it (Goldberger > 1.0: median
0.76, p10 0.35); 0.35-1.0 is reported as doubtful. Checked on GE MAC2000 phone
photos: 0.08 good photo; 1.06 low-res photo with an empty aVL; 1.06 on an
unsupported 6x2 layout. A report is guidance for a human reader, not clinical
validation.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ecg_photo.contracts import QualityLabel, load_manifest

STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
# GE MAC2000 "4x2.5x3_25_R1" / PhysioNet image layout: 12 leads x 2.5 s + II x 10 s
SHORT_LEAD_S = 2.5
RHYTHM_S = 10.0
MIN_COVERAGE = 0.9
FLAT_RANGE_MV = 0.05
RHYTHM_DURATION_TOL = 0.05
IDENTITY_GOOD = 0.35
IDENTITY_BAD = 1.0
MAX_ALIGN_S = 0.04


@dataclass
class LeadQC:
    lead: str
    status: str  # ok | missing | no_signal
    extent_s: float | None = None
    expected_s: float | None = None
    coverage: float | None = None
    range_mV: float | None = None
    flags: list[str] = field(default_factory=list)


@dataclass
class RunQC:
    run_dir: str
    label: QualityLabel
    leads: list[LeadQC]
    einthoven_residual: float | None
    goldberger_residual: float | None
    flags: list[str]
    thresholds: dict[str, float]
    note: str = "guidance only; not clinical validation"


def observed_span(x: np.ndarray, fs: float) -> tuple[np.ndarray, float | None]:
    """Signal cut to its first..last observed (finite) sample, and its start (s)."""
    idx = np.nonzero(np.isfinite(x))[0]
    if len(idx) == 0:
        return x[:0], None
    return x[idx[0] : idx[-1] + 1], idx[0] / fs


def identity_residual(
    parts: list[np.ndarray], target: np.ndarray | None, fs: float
) -> float | None:
    """rms(sum(parts) - target) / rms(target), both mean-removed, over their
    common finite samples, best over target shifts within +-MAX_ALIGN_S (rows
    of a printout are not perfectly aligned). With target None: rms(sum) over
    the mean rms of the parts (an identity summing to zero). None if under
    half of the samples are usable."""
    arrays = parts + ([target] if target is not None else [])
    n = min(len(a) for a in arrays)
    if n < 10:
        return None
    total = np.sum([a[:n] for a in parts], axis=0)
    if target is None:
        m = np.isfinite(total)
        ref = float(np.mean([np.nanstd(a[:n]) for a in parts]))
        if m.sum() < 0.5 * n or not ref > 0:
            return None
        d = total[m] - np.mean(total[m])
        return float(np.sqrt(np.mean(d**2)) / ref)
    best: float | None = None
    max_lag = int(MAX_ALIGN_S * fs)
    for lag in range(-max_lag, max_lag + 1, max(1, max_lag // 20)):
        t = np.roll(target, lag)[:n]
        m = np.isfinite(total) & np.isfinite(t)
        if m.sum() < 0.5 * n:
            continue
        sd = float(np.std(t[m]))
        if not sd > 0:
            continue
        d = (total[m] - np.mean(total[m])) - (t[m] - np.mean(t[m]))
        r = float(np.sqrt(np.mean(d**2)) / sd)
        best = r if best is None or r < best else best
    return best


def run_qc(run_dir: Path) -> RunQC:
    """Quality report of a confirmed run (segments with signals in mV)."""
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir / "manifest.json")
    signals: dict[str, tuple[np.ndarray, float]] = {}
    for seg in manifest.segments:
        if seg.signal_path is None or seg.working_fs_hz is None:
            continue
        x = np.load(run_dir / seg.signal_path, allow_pickle=False).astype(np.float64)
        if seg.observed_mask_path and (run_dir / seg.observed_mask_path).exists():
            obs = np.load(run_dir / seg.observed_mask_path, allow_pickle=False)
            x = np.where(obs, x, np.nan)
        signals[seg.lead_label] = (x, float(seg.working_fs_hz))

    leads: list[LeadQC] = []
    trimmed: dict[str, np.ndarray] = {}
    for name in STANDARD_LEADS:
        if name not in signals:
            leads.append(LeadQC(lead=name, status="missing", flags=["MISSING_LEAD"]))
            continue
        x, fs = signals[name]
        span, _start = observed_span(x, fs)
        if len(span) == 0:
            leads.append(LeadQC(lead=name, status="no_signal", flags=["NO_SIGNAL"]))
            continue
        extent = len(span) / fs
        expected = RHYTHM_S if extent > (SHORT_LEAD_S + RHYTHM_S) / 2 else SHORT_LEAD_S
        finite = span[np.isfinite(span)]
        rng = float(np.percentile(finite, 99.5) - np.percentile(finite, 0.5))
        q = LeadQC(
            lead=name,
            status="ok",
            extent_s=round(extent, 4),
            expected_s=expected,
            coverage=round(float(np.isfinite(span).sum()) / (expected * fs), 4),
            range_mV=round(rng, 4),
        )
        if q.coverage is not None and q.coverage < MIN_COVERAGE:
            q.flags.append("LOW_COVERAGE")
        if rng < FLAT_RANGE_MV:
            q.flags.append("FLAT_TRACE")
        if expected == RHYTHM_S and abs(extent / RHYTHM_S - 1.0) > RHYTHM_DURATION_TOL:
            q.flags.append("RHYTHM_DURATION_MISMATCH")
        leads.append(q)
        trimmed[name] = span

    rates = {r for _x, r in signals.values()}
    common_fs = rates.pop() if len(rates) == 1 else None
    ein = gold = None
    if common_fs is not None:
        if {"I", "II", "III"} <= trimmed.keys():
            # column 1 of the printout: I, II (rhythm strip from t=0), III
            ein = identity_residual([trimmed["I"], trimmed["III"]], trimmed["II"], common_fs)
        if {"aVR", "aVL", "aVF"} <= trimmed.keys():
            parts = [trimmed[k] for k in ("aVR", "aVL", "aVF")]
            gold = identity_residual(parts, None, common_fs)

    flags = sorted({f for q in leads for f in q.flags})
    worst = max((v for v in (ein, gold) if v is not None), default=None)
    if worst is not None and worst > IDENTITY_BAD:
        flags.append("LIMB_LEADS_INCONSISTENT")
    elif worst is not None and worst > IDENTITY_GOOD:
        flags.append("LIMB_LEADS_DOUBTFUL")
    hard = {"MISSING_LEAD", "NO_SIGNAL", "FLAT_TRACE", "LIMB_LEADS_INCONSISTENT"}
    if hard & set(flags):
        label = QualityLabel.insufficient
    elif flags:
        label = QualityLabel.acceptable
    else:
        label = QualityLabel.good
    return RunQC(
        run_dir=str(run_dir),
        label=label,
        leads=leads,
        einthoven_residual=None if ein is None else round(ein, 4),
        goldberger_residual=None if gold is None else round(gold, 4),
        flags=flags,
        thresholds={
            "min_coverage": MIN_COVERAGE,
            "flat_range_mV": FLAT_RANGE_MV,
            "rhythm_duration_tol": RHYTHM_DURATION_TOL,
            "identity_good": IDENTITY_GOOD,
            "identity_bad": IDENTITY_BAD,
        },
    )


def qc_json(report: RunQC) -> str:
    d = asdict(report)
    d["label"] = report.label.value
    return json.dumps(d, indent=2, ensure_ascii=False)
