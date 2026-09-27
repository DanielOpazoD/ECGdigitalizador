"""Global ECG intervals (PR, QRS, QT, QTc) from a confirmed run (O7 in
docs/mission.md).

Per lead: QRS complexes (qc.detect_qrs) -> median beat -> fiducials
(P onset, QRS onset/offset, T end) -> intervals. The reported value of each
interval is the median over the leads where it was found, and its spread
(IQR over leads) is reported as the uncertainty together with the time
resolution of the signal. An interval that cannot be found is `None` with a
reason, never a default.

Fiducials on the median beat, relative to an isoelectric level taken on the
PR segment:
- QRS onset/offset: where the slope energy (|derivative|, 8 ms smoothing)
  stays below a fraction of its QRS maximum for a sustained 12 ms, searching
  out from the R peak;
- T end: tangent at the steepest descent of the T wave after its peak,
  intersected with the isoelectric level (tangent method);
- P onset: tangent at the steepest rise of the P wave before its peak,
  intersected with the isoelectric level; no P wave (below 0.04 mV, e.g.
  atrial fibrillation) -> PR not measurable.

Parameters were tuned on the odd LUDB records and are reported on the even
ones (docs/evaluation.md, F9). Guidance only; not clinical validation.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ecg_photo.contracts import load_manifest
from ecg_photo.qc import STANDARD_LEADS, detect_qrs, observed_span

PRE_S = 0.35  # median-beat window before R
POST_S = 0.65  # and after R (capped at 0.9 x RR)
SMOOTH_S = 0.008
QRS_SEARCH_S = 0.14  # QRS boundaries searched within +-140 ms of R
QRS_ENERGY_FRAC = 0.04
SUSTAIN_S = 0.012
MIN_P_MV = 0.04
MIN_T_MV = 0.05
T_SEARCH_RR = 0.6  # T peak searched up to 0.6 x RR after QRS onset
T_SEARCH_MAX_S = 0.6
T_END_FRAC: float | None = 0.15  # T end: back within 15 % of |T amplitude|; None: tangent
P_ON_FRAC: float | None = None
QTC_RR_RANGE_S = (0.3, 2.0)
# an interval is "doubtful" when found in fewer than MIN_LEADS leads or when
# its IQR over leads exceeds MAX_SPREAD_MS (chosen on the odd LUDB records:
# flagged values had about twice the error of the kept ones)
MIN_LEADS = 3
MAX_SPREAD_MS = {"pr_ms": 40.0, "qrs_ms": 30.0, "qt_ms": 40.0}
# percentile over leads reported for each interval. QT is a global interval
# (earliest QRS onset to latest T end): the median over leads read 19-22 ms
# shorter than the GE 12SL QT a GE electrocardiograph prints; the 75th
# percentile has no bias against it (F11, PTB-XL+). PR and QRS: the median
# agrees with 12SL within a few ms.
AGG_PERCENTILE = {"pr_ms": 50.0, "qrs_ms": 50.0, "qt_ms": 75.0}


@dataclass
class LeadIntervals:
    lead: str
    n_beats: int
    pr_ms: float | None = None
    qrs_ms: float | None = None
    qt_ms: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Intervals:
    rr_ms: float | None
    hr_bpm: float | None
    pr_ms: float | None
    qrs_ms: float | None
    qt_ms: float | None
    qtc_bazett_ms: float | None
    qtc_fridericia_ms: float | None
    spread_ms: dict[str, float | None]  # IQR over leads per interval
    status: dict[str, str]  # ok | doubtful | unavailable, per interval
    n_leads: dict[str, int]
    resolution_ms: float | None
    reasons: dict[str, str]
    leads: list[LeadIntervals]
    # comparison with the values printed by the electrocardiograph, per interval
    printed: dict[str, dict] = field(default_factory=dict)
    note: str = "guidance only; not clinical validation"


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def median_beat(x: np.ndarray, fs: float, beats: np.ndarray, rr_s: float) -> np.ndarray | None:
    """Median of the beat windows around each R that fit in the signal and
    contain no gap; the R peak sits at round(PRE_S * fs)."""
    pre = round(PRE_S * fs)
    post = round(min(POST_S, 0.9 * rr_s) * fs)
    wins = [
        x[b - pre : b + post]
        for b in beats
        if b - pre >= 0 and b + post <= len(x) and np.isfinite(x[b - pre : b + post]).all()
    ]
    if not wins:
        return None
    return np.median(np.stack(wins), axis=0)


def _sustained_below(
    e: np.ndarray, thr: float, start: int, stop: int, step: int, n: int
) -> int | None:
    """First index from `start` towards `stop` (exclusive) where e stays below
    thr for n samples."""
    run = 0
    for i in range(start, stop, step):
        run = run + 1 if e[i] < thr else 0
        if run >= n:
            return i - step * (n - 1)
    return None


def _tangent_to_level(b: np.ndarray, i: int, level: float) -> float | None:
    """Index where the tangent of b at i crosses `level` (fractional)."""
    slope = (b[min(i + 1, len(b) - 1)] - b[max(i - 1, 0)]) / 2.0
    if slope == 0:
        return None
    return i + (level - b[i]) / slope


def _return_to_level(
    b: np.ndarray, i: int, level: float, tol: float, step: int, stop: int
) -> float | None:
    """First index from i (towards `stop`) where b is within tol of level."""
    stop = max(0, min(len(b) - 1, stop))
    for j in range(i, stop, step):
        if abs(b[j] - level) <= tol:
            return float(j)
    return None


def beat_fiducials(b: np.ndarray, fs: float, rr_s: float) -> dict[str, float | None]:
    """Fiducials (sample index in the median beat) of one median beat."""
    r = round(PRE_S * fs)
    d = np.gradient(_smooth(b, max(1, round(SMOOTH_S * fs))))
    e = np.abs(d)
    w = round(QRS_SEARCH_S * fs)
    lo, hi = max(1, r - w), min(len(b) - 1, r + w)
    thr = QRS_ENERGY_FRAC * float(e[lo:hi].max())
    n = max(2, round(SUSTAIN_S * fs))
    q_on = _sustained_below(e, thr, r, lo - 1, -1, n)
    q_off = _sustained_below(e, thr, r, hi, 1, n)
    out: dict[str, float | None] = {"qrs_on": q_on, "qrs_off": q_off, "p_on": None, "t_end": None}
    if q_on is None or q_off is None:
        return out
    # isoelectric level: 20 ms just before QRS onset (PR segment)
    iso = float(np.median(b[max(0, q_on - round(0.02 * fs)) : q_on + 1]))

    # T wave: largest deviation from QRS offset + 60 ms up to T_SEARCH_RR x RR
    # after QRS onset (beyond it lies the next P wave)
    t_lo = q_off + round(0.06 * fs)
    t_hi = min(len(b), q_on + round(min(T_SEARCH_MAX_S, T_SEARCH_RR * rr_s) * fs))
    if t_lo < t_hi - 3:
        seg = b[t_lo:t_hi] - iso
        tp = t_lo + int(np.argmax(np.abs(seg)))
        t_amp = b[tp] - iso
        if abs(t_amp) >= MIN_T_MV and tp < len(b) - 2:
            ds = d[tp : min(len(b), tp + round(0.2 * fs))]
            # steepest return towards the isoelectric level after the peak
            k = tp + int(np.argmin(ds) if t_amp > 0 else np.argmax(ds))
            t_end = (
                _tangent_to_level(b, k, iso)
                if T_END_FRAC is None
                else _return_to_level(b, k, iso, T_END_FRAC * abs(t_amp), +1, t_hi)
            )
            if t_end is not None and tp < t_end <= len(b) - 1:
                out["t_end"] = t_end

    # P wave: largest deviation 30-300 ms before QRS onset
    p_lo, p_hi = max(1, q_on - round(0.30 * fs)), q_on - round(0.03 * fs)
    if p_hi - p_lo > 3:
        seg = b[p_lo:p_hi] - iso
        pp = p_lo + int(np.argmax(np.abs(seg)))
        p_amp = b[pp] - iso
        if abs(p_amp) >= MIN_P_MV and pp > p_lo:
            du = d[p_lo:pp]
            k = p_lo + int(np.argmax(du) if p_amp > 0 else np.argmin(du))
            p_on = (
                _tangent_to_level(b, k, iso)
                if P_ON_FRAC is None
                else _return_to_level(
                    b, k, iso, P_ON_FRAC * abs(p_amp), -1, p_lo - round(0.05 * fs)
                )
            )
            if p_on is not None and p_lo - round(0.05 * fs) <= p_on < pp:
                out["p_on"] = p_on
    return out


def lead_intervals(name: str, x: np.ndarray, fs: float) -> LeadIntervals:
    span, _ = observed_span(x, fs)
    beats = detect_qrs(span, fs) if len(span) else np.array([], int)
    li = LeadIntervals(lead=name, n_beats=len(beats))
    if len(beats) < 2:
        li.notes.append("fewer than 2 beats")
        return li
    rr_s = float(np.median(np.diff(beats))) / fs
    b = median_beat(span, fs, beats, rr_s)
    if b is None:
        li.notes.append("no complete beat window")
        return li
    f = beat_fiducials(b, fs, rr_s)
    ms = 1000.0 / fs
    if f["qrs_on"] is not None and f["qrs_off"] is not None:
        li.qrs_ms = (f["qrs_off"] - f["qrs_on"]) * ms
        if f["t_end"] is not None:
            li.qt_ms = (f["t_end"] - f["qrs_on"]) * ms
        else:
            li.notes.append("T end not found")
        if f["p_on"] is not None:
            li.pr_ms = (f["qrs_on"] - f["p_on"]) * ms
        else:
            li.notes.append("P wave not found")
    else:
        li.notes.append("QRS boundaries not found")
    return li


def measure_signals(
    signals: dict[str, tuple[np.ndarray, float]], rr_ms: float | None = None
) -> Intervals:
    """Intervals from lead -> (signal in mV, fs). `rr_ms` (e.g. the QC's
    trimmed-mean RR of the rhythm strip) is used for heart rate and QTc."""
    leads = [lead_intervals(n, *signals[n]) for n in STANDARD_LEADS if n in signals]
    res: dict[str, float | None] = {}
    spread: dict[str, float | None] = {}
    count: dict[str, int] = {}
    reasons: dict[str, str] = {}
    status: dict[str, str] = {}
    for key in ("pr_ms", "qrs_ms", "qt_ms"):
        vals = [getattr(li, key) for li in leads if getattr(li, key) is not None]
        count[key] = len(vals)
        if vals:
            res[key] = float(np.percentile(vals, AGG_PERCENTILE[key]))
            iqr = float(np.subtract(*np.percentile(vals, [75, 25])))
            spread[key] = iqr
            if len(vals) < MIN_LEADS:
                status[key] = "doubtful"
                reasons[key] = f"found in {len(vals)} lead(s) only"
            elif iqr > MAX_SPREAD_MS[key]:
                status[key] = "doubtful"
                reasons[key] = f"leads disagree (IQR {iqr:.0f} ms > {MAX_SPREAD_MS[key]:.0f})"
            else:
                status[key] = "ok"
        else:
            res[key] = spread[key] = None
            status[key] = "unavailable"
            reasons[key] = "not found in any lead"
    qtc_b = qtc_f = None
    if res["qt_ms"] is not None and rr_ms is not None:
        rr_s = rr_ms / 1000.0
        if QTC_RR_RANGE_S[0] <= rr_s <= QTC_RR_RANGE_S[1]:
            qtc_b = res["qt_ms"] / rr_s**0.5
            qtc_f = res["qt_ms"] / rr_s ** (1 / 3)
        else:
            reasons["qtc"] = f"RR {rr_ms:.0f} ms outside {QTC_RR_RANGE_S} s"
    elif res["qt_ms"] is not None:
        reasons["qtc"] = "RR not measured"
    fss = {fs for _x, fs in signals.values()}
    return Intervals(
        rr_ms=rr_ms,
        hr_bpm=None if rr_ms is None else 60000.0 / rr_ms,
        pr_ms=res["pr_ms"],
        qrs_ms=res["qrs_ms"],
        qt_ms=res["qt_ms"],
        qtc_bazett_ms=qtc_b,
        qtc_fridericia_ms=qtc_f,
        spread_ms=spread,
        status=status,
        n_leads=count,
        resolution_ms=1000.0 / min(fss) if fss else None,
        reasons=reasons,
        leads=leads,
    )


def run_signals(run_dir: Path) -> dict[str, tuple[np.ndarray, float]]:
    """Lead -> (signal in mV with NaN where not observed, fs) of a confirmed run."""
    manifest = load_manifest(Path(run_dir) / "manifest.json")
    out: dict[str, tuple[np.ndarray, float]] = {}
    for seg in manifest.segments:
        if seg.signal_path is None or seg.working_fs_hz is None or seg.units != "mV":
            continue
        x = np.load(Path(run_dir) / seg.signal_path, allow_pickle=False).astype(np.float64)
        if seg.observed_mask_path and (Path(run_dir) / seg.observed_mask_path).exists():
            x = np.where(np.load(Path(run_dir) / seg.observed_mask_path), x, np.nan)
        out[seg.lead_label] = (x, float(seg.working_fs_hz))
    return out


def measure_run(run_dir: Path, rr_ms: float | None = None) -> Intervals:
    return measure_signals(run_signals(run_dir), rr_ms=rr_ms)


def intervals_json(iv: Intervals) -> dict:
    d = asdict(iv)
    for k, v in list(d.items()):
        if isinstance(v, float):
            d[k] = round(v, 1)
    return d


INTERVALS_FILENAME = "intervals.json"


def demote_on_qc(iv: Intervals, qc_label: str | None) -> Intervals:
    """`ok` intervals become `doubtful` when the digitization's quality label
    is `insufficient`."""
    if qc_label != "insufficient":
        return iv
    for key, st in iv.status.items():
        if st == "ok":
            iv.status[key] = "doubtful"
            iv.reasons[key] = "digitization quality insufficient"
    return iv


# largest |ours - printed| still counted as agreement (ms): 95th percentile of
# the difference against the GE 12SL values over `ok` measurements on the
# first half of the F11 PTB-XL sample (printed 3x4 + II signal)
PRINTED_TOL_MS = {"pr_ms": 0.0, "qrs_ms": 0.0, "qt_ms": 0.0, "qtc_bazett_ms": 0.0}


def compare_printed(iv: Intervals, printed: dict[str, float | None]) -> Intervals:
    """Compare with the electrocardiograph's printed values (keys of
    PRINTED_TOL_MS; None/absent = not printed). An `ok` interval outside the
    tolerance becomes `doubtful`: either the digitization or the machine is
    wrong, and the original must be looked at."""
    for key, tol in PRINTED_TOL_MS.items():
        p = printed.get(key)
        if p is None:
            continue
        if not (np.isfinite(p) and p > 0):
            raise ValueError(f"printed {key} must be > 0")
        ours = getattr(iv, key)
        entry: dict = {"printed_ms": float(p), "tolerance_ms": tol}
        if ours is None:
            entry["agrees"] = None
        else:
            entry["diff_ms"] = round(ours - p, 1)
            entry["agrees"] = bool(abs(ours - p) <= tol)
            status_key = "qt_ms" if key == "qtc_bazett_ms" else key
            if not entry["agrees"] and iv.status.get(status_key) == "ok":
                iv.status[status_key] = "doubtful"
                iv.reasons[status_key] = (
                    f"differs from printed {key} by {ours - p:+.0f} ms (> {tol:.0f})"
                )
        iv.printed[key] = entry
    return iv


def write_intervals_report(run_dir: Path, printed: dict[str, float | None] | None = None) -> dict:
    """Write `intervals.json` next to a confirmed run's manifest, using the RR
    of its quality report (qc.json) when present. Never raises: a failure is
    written as {"error": ...} so its absence is explicit. When the quality
    report says `insufficient`, no interval is `ok` (F10: those images had
    about twice the interval error)."""
    run_dir = Path(run_dir)
    try:
        rr = None
        qc_label = None
        qc_path = run_dir / "qc.json"
        if qc_path.exists():
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            rr = qc.get("rr_measured_ms")
            qc_label = qc.get("label")
        iv = demote_on_qc(measure_run(run_dir, rr_ms=rr), qc_label)
        d = intervals_json(compare_printed(iv, printed or {}))
    except Exception as e:  # noqa: BLE001 - measurements are advisory, the run stands
        d = {"error": f"{type(e).__name__}: {e}"[:300]}
    (run_dir / INTERVALS_FILENAME).write_text(
        json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return d


def read_intervals_report(run_dir: Path) -> dict | None:
    p = Path(run_dir) / INTERVALS_FILENAME
    if not p.exists():
        return None
    result: dict = json.loads(p.read_text(encoding="utf-8"))
    return result
