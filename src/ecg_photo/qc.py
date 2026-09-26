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

Optionally the RR interval measured on the digitized rhythm strip is compared
with the one the electrocardiograph printed (entered by the user: RR in ms or
heart rate in bpm): a digitization whose time axis disagrees by more than
RR_TOL is not usable for timing.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d  # type: ignore[import-untyped]
from scipy.signal import butter, filtfilt, find_peaks  # type: ignore[import-untyped]

from ecg_photo.contracts import Manifest, QualityLabel, load_manifest

STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
# GE MAC2000 "4x2.5x3_25_R1" / PhysioNet image layout: 12 leads x 2.5 s + II x 10 s
SHORT_LEAD_S = 2.5
RHYTHM_S = 10.0
MIN_COVERAGE = 0.9
FLAT_RANGE_MV = 0.05
RHYTHM_DURATION_TOL = 0.05
IDENTITY_GOOD = 0.35
IDENTITY_BAD = 1.0
# absolute floor (mV) for the rms that normalises the identity residuals; 0
# keeps the pure relative residual (see F7 paso 7 in docs/evaluation.md)
IDENTITY_RMS_FLOOR_MV = 0.0
MAX_ALIGN_S = 0.04
RR_TOL = 0.05  # vs printed RR; printed HR is an integer (+-1 bpm ~ 1-2 %)
MIN_RR_S = 0.2  # 300 bpm refractory: a 0.3 s limit halved an SVT at 207 bpm (RR 290 ms)
# QRS detector (Pan-Tompkins-like): slope energy of the 5-25 Hz band over 100 ms,
# amplitude-scaled (sqrt); candidates above QRS_FLOOR x p99.5, beats above
# QRS_KEEP x the candidates' median. On 4 GE MAC2000 photos (sinus with 0.9 mV
# baseline drift, 2:1 AV block with tall T, SVT RR 290 ms, AF with aberrant
# beats) the beat count was exact for floors 0.25-0.35 and windows 80-150 ms.
QRS_BAND_HZ = (5.0, 25.0)
QRS_WINDOW_S = 0.1
QRS_FLOOR = 0.3
QRS_KEEP = 0.4
# RR intervals kept for the mean, as fractions of the median RR
RR_KEEP = (0.5, 1.8)
MIN_RR_STRIP_S = 5.0  # shortest printed lead used for RR when there is no 10 s strip


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
    n_beats: int | None = None
    rr_measured_ms: float | None = None  # mean RR of the rhythm strip
    rr_median_ms: float | None = None
    rr_printed_ms: float | None = None
    rr_error_pct: float | None = None
    rr_lead: str | None = None
    layout: str | None = None  # engine-reported page layout, if any
    note: str = "guidance only; not clinical validation"


_LAYOUT_RE = re.compile(r"engine layout '([^']+)'")


def engine_layout(manifest: Manifest) -> str | None:
    """Page layout reported by the engine (kept in each segment's
    representation_evidence by digitize_page), or None."""
    for seg in manifest.segments:
        m = _LAYOUT_RE.search(seg.representation_evidence or "")
        if m:
            return m.group(1)
    return None


def short_lead_s(layout: str | None) -> float:
    """Printed duration of a non-rhythm lead: the 10 s page split into the
    layout's columns (3x4 -> 2.5 s, 6x2 -> 5 s, 12x1 -> 10 s); 2.5 s (3x4, the
    GE MAC2000 format) when the layout is unknown."""
    m = re.search(r"(\d+)x(\d+)", layout or "")
    if m and int(m.group(2)) > 0:
        return RHYTHM_S / int(m.group(2))
    return SHORT_LEAD_S


def observed_span(x: np.ndarray, fs: float) -> tuple[np.ndarray, float | None]:
    """Signal cut to its first..last observed (finite) sample, and its start (s)."""
    idx = np.nonzero(np.isfinite(x))[0]
    if len(idx) == 0:
        return x[:0], None
    return x[idx[0] : idx[-1] + 1], idx[0] / fs


def detect_qrs(x: np.ndarray, fs: float) -> np.ndarray:
    """Sample indices of QRS complexes on a rhythm strip.

    Slope energy of the QRS band (steep QRS; slow P/T waves and baseline
    drift from curved paper stay low), smoothed over QRS_WINDOW_S and
    amplitude-scaled; peaks >= MIN_RR_S apart above QRS_FLOOR x p99.5 are
    candidates, and those above QRS_KEEP x the candidates' median are beats
    (the median, unlike the maximum, is not moved by one large aberrant beat).
    Unobserved samples (NaN) are bridged for filtering only and never become
    beats. The index is the energy peak (within ~50 ms of the QRS)."""
    v = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < fs:
        return np.zeros(0, dtype=int)
    idx = np.arange(len(v))
    v = np.interp(idx, idx[ok], v[ok])
    nyq = fs / 2.0
    lo, hi = QRS_BAND_HZ[0] / nyq, min(QRS_BAND_HZ[1], 0.8 * nyq) / nyq
    if not 0.0 < lo < hi < 1.0 or len(v) <= 15:
        return np.zeros(0, dtype=int)
    b, a = butter(2, [lo, hi], btype="band")
    slope = np.gradient(filtfilt(b, a, v))
    energy = np.sqrt(uniform_filter1d(slope**2, size=max(1, int(QRS_WINDOW_S * fs))))
    energy[~ok] = 0.0
    floor = QRS_FLOOR * float(np.percentile(energy, 99.5))
    if not floor > 0:
        return np.zeros(0, dtype=int)
    cand, props = find_peaks(energy, height=floor, distance=max(1, int(MIN_RR_S * fs)))
    if len(cand) == 0:
        return np.zeros(0, dtype=int)
    heights = props["peak_heights"]
    return np.asarray(cand[heights >= QRS_KEEP * float(np.median(heights))], dtype=int)


def identity_residual(
    parts: list[np.ndarray], target: np.ndarray | None, fs: float, floor_mV: float = 0.0
) -> float | None:
    """rms(sum(parts) - target) / rms(target), both mean-removed, over their
    common finite samples, best over target shifts within +-MAX_ALIGN_S (rows
    of a printout are not perfectly aligned). With target None: rms(sum) over
    the mean rms of the parts (an identity summing to zero). The normalising
    rms is at least floor_mV, so low-voltage leads are judged by an absolute
    error. None if under half of the samples are usable."""
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
        return float(np.sqrt(np.mean(d**2)) / max(ref, floor_mV))
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
        r = float(np.sqrt(np.mean(d**2)) / max(sd, floor_mV))
        best = r if best is None or r < best else best
    return best


def run_qc(
    run_dir: Path,
    *,
    printed_rr_ms: float | None = None,
    printed_hr_bpm: float | None = None,
    identity_floor_mV: float | None = None,
) -> RunQC:
    """Quality report of a confirmed run (segments with signals in mV).
    printed_rr_ms / printed_hr_bpm: values printed by the electrocardiograph
    (RR preferred when both are given), compared with the rhythm strip."""
    if printed_rr_ms is None and printed_hr_bpm is not None:
        if not printed_hr_bpm > 0:
            raise ValueError("printed_hr_bpm must be > 0")
        printed_rr_ms = 60000.0 / printed_hr_bpm
    if printed_rr_ms is not None and not printed_rr_ms > 0:
        raise ValueError("printed_rr_ms must be > 0")
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir / "manifest.json")
    layout = engine_layout(manifest)
    short_s = short_lead_s(layout)
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
        expected = RHYTHM_S if extent > (short_s + RHYTHM_S) / 2 else short_s
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

    floor = IDENTITY_RMS_FLOOR_MV if identity_floor_mV is None else identity_floor_mV
    rates = {r for _x, r in signals.values()}
    common_fs = rates.pop() if len(rates) == 1 else None
    ein = gold = None
    if common_fs is not None:
        if {"I", "II", "III"} <= trimmed.keys():
            # column 1 of the printout: I, II (rhythm strip from t=0), III
            ein = identity_residual([trimmed["I"], trimmed["III"]], trimmed["II"], common_fs, floor)
        if {"aVR", "aVL", "aVF"} <= trimmed.keys():
            parts = [trimmed[k] for k in ("aVR", "aVL", "aVF")]
            gold = identity_residual(parts, None, common_fs, floor)

    n_beats = rr_ms = rr_err = rr_median = None
    rhythm = [q.lead for q in leads if q.status == "ok" and q.expected_s == RHYTHM_S]
    if not rhythm:
        # no 10 s strip (e.g. 6x2: every lead 5 s): the longest printed
        # leads, II first, still hold several beats
        rhythm = sorted(
            (q.lead for q in leads if q.status == "ok" and (q.expected_s or 0) >= MIN_RR_STRIP_S),
            key=lambda n: n != "II",
        )
    rr_lead = rhythm[0] if rhythm else None
    if rhythm:
        x, fs = signals[rhythm[0]]
        beats = detect_qrs(observed_span(x, fs)[0], fs)
        n_beats = len(beats)
        if n_beats >= 3:
            rr = np.diff(beats) / fs * 1000.0
            # the electrocardiograph prints the mean RR; in an irregular rhythm
            # (AF) the median differs from it by more than the tolerance.
            # Intervals far from the median are a missed beat (~2x) or a
            # spurious one (split interval) of the digitized trace, not rhythm:
            # they are left out of the mean (F7 Kaggle: strips > 5 % off the truth
            # 7/104 -> 1/104).
            rr_median = float(np.median(rr))
            keep = rr[(rr >= RR_KEEP[0] * rr_median) & (rr <= RR_KEEP[1] * rr_median)]
            rr_ms = float(np.mean(keep if len(keep) else rr))

    flags = sorted({f for q in leads for f in q.flags})
    if printed_rr_ms is not None:
        if rr_ms is None:
            flags.append("RR_NOT_MEASURABLE")
        else:
            rr_err = 100.0 * (rr_ms / printed_rr_ms - 1.0)
            if abs(rr_err) > 100.0 * RR_TOL:
                flags.append("RR_MISMATCH_PRINTED")
    worst = max((v for v in (ein, gold) if v is not None), default=None)
    if worst is not None and worst > IDENTITY_BAD:
        flags.append("LIMB_LEADS_INCONSISTENT")
    elif worst is not None and worst > IDENTITY_GOOD:
        flags.append("LIMB_LEADS_DOUBTFUL")
    hard = {
        "MISSING_LEAD",
        "NO_SIGNAL",
        "FLAT_TRACE",
        "LIMB_LEADS_INCONSISTENT",
        "RR_MISMATCH_PRINTED",
    }
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
            "identity_rms_floor_mV": floor,
            "rr_tol": RR_TOL,
            "min_rr_s": MIN_RR_S,
            "qrs_floor": QRS_FLOOR,
            "qrs_keep": QRS_KEEP,
            "rr_keep_min": RR_KEEP[0],
            "rr_keep_max": RR_KEEP[1],
        },
        n_beats=n_beats,
        rr_measured_ms=None if rr_ms is None else round(rr_ms, 1),
        rr_median_ms=None if rr_median is None else round(rr_median, 1),
        rr_printed_ms=None if printed_rr_ms is None else round(printed_rr_ms, 1),
        rr_error_pct=None if rr_err is None else round(rr_err, 2),
        rr_lead=rr_lead,
        layout=layout,
    )


def qc_json(report: RunQC) -> str:
    d = asdict(report)
    d["label"] = report.label.value
    return json.dumps(d, indent=2, ensure_ascii=False)


QC_FILENAME = "qc.json"


def write_qc_report(run_dir: Path) -> dict:
    """Write `qc.json` next to the run's manifest and return it as a dict. A
    report that cannot be computed is written as {"label": null, "error": ...}
    so its absence is explicit; never raises."""
    run_dir = Path(run_dir)
    try:
        text = qc_json(run_qc(run_dir))
    except Exception as e:  # noqa: BLE001 - QC is advisory, the run stands
        text = json.dumps({"label": None, "error": f"{type(e).__name__}: {e}"[:300]})
    (run_dir / QC_FILENAME).write_text(text + "\n", encoding="utf-8")
    result: dict = json.loads(text)
    return result


def read_qc_report(run_dir: Path) -> dict | None:
    path = Path(run_dir) / QC_FILENAME
    if not path.exists():
        return None
    result: dict = json.loads(path.read_text(encoding="utf-8"))
    return result
