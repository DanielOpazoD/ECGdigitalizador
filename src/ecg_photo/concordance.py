"""Concordance between the program and the values the electrocardiograph
printed, over a folder of photos (validation protocol:
docs/protocolo_validacion.md).

Input: the photos and a CSV with what each sheet header shows (one row per
file; `;` or `,` separated, decimal comma accepted, blank = not printed).
Each photo goes through the normal one-command chain (process_file) with its
printed values; the result is a per-photo table and, per measurement, the
Bland-Altman statistics (bias, SD, 95 % limits of agreement) and the share
of values within the F11/F12 tolerance, for all values and for `ok` ones.
Resumable: a photo whose summary.json already exists is not processed again.
"""

import csv
import io
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ecg_photo.digitizers.base import Digitizer
from ecg_photo.ingest import IngestRejected
from ecg_photo.intervals import PRINTED_TOL_DEG, PRINTED_TOL_MS, write_intervals_report
from ecg_photo.printed import PrintedValues
from ecg_photo.process import ProcessOptions, process_file
from ecg_photo.qc import RR_TOL

CSV_COLUMNS = ("file", "hr_bpm", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms", "axis_deg")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".pdf")
# measurement -> (key in the intervals summary, printed field, unit, tolerance)
MEASURES = {
    "FC": ("hr_bpm", "hr_bpm", "lpm", None),  # tolerance: RR within qc.RR_TOL
    "PR": ("pr_ms", "pr_ms", "ms", PRINTED_TOL_MS["pr_ms"]),
    "QRS": ("qrs_ms", "qrs_ms", "ms", PRINTED_TOL_MS["qrs_ms"]),
    "QT": ("qt_ms", "qt_ms", "ms", PRINTED_TOL_MS["qt_ms"]),
    "QTc": ("qtc_bazett_ms", "qtc_ms", "ms", PRINTED_TOL_MS["qtc_bazett_ms"]),
    "Eje": ("qrs_axis_deg", "axis_deg", "°", PRINTED_TOL_DEG["qrs_axis_deg"]),
}
STATUS_KEY = {"PR": "pr_ms", "QRS": "qrs_ms", "QT": "qt_ms", "QTc": "qt_ms", "Eje": "qrs_axis_deg"}
PRINTED_KEY = {
    "PR": "pr_ms",
    "QRS": "qrs_ms",
    "QT": "qt_ms",
    "QTc": "qtc_bazett_ms",
    "Eje": "qrs_axis_deg",
}


def _num(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    return float(text.replace(",", "."))


def read_printed_csv(path: Path) -> dict[str, PrintedValues]:
    """file name -> printed values. Raises ValueError naming the bad row."""
    raw = Path(path).read_text(encoding="utf-8-sig")
    delim = ";" if raw.splitlines()[0].count(";") >= raw.splitlines()[0].count(",") else ","
    out: dict[str, PrintedValues] = {}
    for n, row in enumerate(csv.DictReader(io.StringIO(raw), delimiter=delim), start=2):
        name = (row.get("file") or "").strip()
        if not name:
            continue
        try:
            values = {k: _num(row.get(k) or "") for k in CSV_COLUMNS[1:]}
            out[name] = PrintedValues(**values)
        except ValueError as e:
            raise ValueError(f"row {n} ({name}): {e}") from e
    return out


def write_template(photos: list[Path], path: Path) -> None:
    """Printed-values CSV with one empty row per photo, to fill in by hand."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(CSV_COLUMNS)
        for p in photos:
            w.writerow([p.name] + [""] * (len(CSV_COLUMNS) - 1))


def list_photos(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


@dataclass
class Row:
    file: str
    status: str  # processed | rejected | error | no_printed_values
    qc_label: str | None = None
    ours: dict[str, float | None] = field(default_factory=dict)
    printed: dict[str, float | None] = field(default_factory=dict)
    measure_status: dict[str, str | None] = field(default_factory=dict)
    error: str | None = None


def _options(base: ProcessOptions, pv: PrintedValues) -> ProcessOptions:
    return replace(
        base,
        printed_rr_ms=pv.rr_ms,
        printed_hr_bpm=pv.hr_bpm,
        printed_intervals_ms=pv.interval_keys(),
    )


def _row_from_summary(name: str, summary: dict, pv: PrintedValues) -> Row:
    iv = summary.get("intervals") or {}
    status = iv.get("status") or {}
    printed_hr = pv.hr_bpm if pv.hr_bpm is not None else (60000.0 / pv.rr_ms if pv.rr_ms else None)
    row = Row(file=name, status="processed", qc_label=(summary.get("qc") or {}).get("label"))
    for label, (key, pkey, _unit, _tol) in MEASURES.items():
        row.ours[label] = iv.get(key)
        row.printed[label] = printed_hr if label == "FC" else getattr(pv, pkey)
        if label not in STATUS_KEY:
            row.measure_status[label] = None
            continue
        # the measurement's own status, before the comparison with the printed
        # value demoted it: counting only values that already agree would be
        # circular
        cmp = (iv.get("printed") or {}).get(PRINTED_KEY[label]) or {}
        row.measure_status[label] = cmp.get("status_before", status.get(STATUS_KEY[label]))
    return row


def run_concordance(
    folder: Path,
    printed: dict[str, PrintedValues],
    out: Path,
    digitizer: Digitizer,
    base: ProcessOptions,
) -> list[Row]:
    rows: list[Row] = []
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for photo in list_photos(folder):
        pv = printed.get(photo.name)
        if pv is None:
            rows.append(Row(file=photo.name, status="no_printed_values"))
            continue
        target = out / "runs" / photo.stem
        summary_path = target / "summary.json"
        try:
            if summary_path.exists():
                # resume: the engine is not run again, but the intervals are
                # re-measured from the saved run with the current code
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                run_dir = target / summary["run_dirs"]["confirmed"]
                iv = write_intervals_report(run_dir, printed=pv.interval_keys())
                summary["intervals"] = {k: v for k, v in iv.items() if k != "leads"}
            else:
                if target.exists():  # an interrupted attempt: start it again
                    import shutil

                    shutil.rmtree(target)
                summary = process_file(photo, target, digitizer, _options(base, pv))
            rows.append(_row_from_summary(photo.name, summary, pv))
        except IngestRejected as e:
            rows.append(Row(file=photo.name, status="rejected", error=str(e.reason_code)))
        except (ValueError, RuntimeError, OSError) as e:
            rows.append(
                Row(file=photo.name, status="error", error=f"{type(e).__name__}: {e}"[:300])
            )
    return rows


def _pairs(rows: list[Row], label: str, only_ok: bool) -> list[tuple[float, float, str | None]]:
    """(ours, printed, status) of the processed photos where both exist."""
    out = []
    for r in rows:
        o, p = r.ours.get(label), r.printed.get(label)
        if r.status != "processed" or o is None or p is None:
            continue
        st = r.measure_status.get(label)
        if only_ok and label != "FC" and st != "ok":
            continue
        out.append((float(o), float(p), st))
    return out


def _diff(label: str, ours: float, printed: float) -> float:
    if label == "Eje":
        return (ours - printed + 180.0) % 360.0 - 180.0
    return ours - printed


def _within(label: str, ours: float, printed: float) -> bool:
    if label == "FC":  # the RR tolerance of the quality check, as a relative error
        return abs(ours / printed - 1.0) <= RR_TOL
    tol = MEASURES[label][3]
    assert tol is not None
    return abs(_diff(label, ours, printed)) <= tol


def summarize(rows: list[Row]) -> dict:
    """Per measurement: Bland-Altman statistics over all compared values and
    over `ok` ones (FC: every measured value)."""
    out: dict = {
        "photos": len(rows),
        "processed": sum(r.status == "processed" for r in rows),
        "not_processed": {r.file: r.error or r.status for r in rows if r.status != "processed"},
        "measures": {},
        "note": "guidance only; not clinical validation",
    }
    for label, (_key, _pkey, unit, tol) in MEASURES.items():
        res: dict = {"unit": unit, "tolerance": f"{100 * RR_TOL:.0f} %" if label == "FC" else tol}
        for subset in ("all", "ok"):
            pairs = [(o, p) for o, p, _ in _pairs(rows, label, subset == "ok")]
            printed_n = sum(
                1 for r in rows if r.status == "processed" and r.printed.get(label) is not None
            )
            if not pairs:
                res[subset] = {"n": 0, "printed": printed_n}
                continue
            d = np.array([_diff(label, o, p) for o, p in pairs])
            sd = float(d.std(ddof=1)) if len(d) > 1 else None
            res[subset] = {
                "n": len(d),
                "printed": printed_n,
                "bias": float(d.mean()),
                "sd": sd,
                "loa": [float(d.mean() - 1.96 * sd), float(d.mean() + 1.96 * sd)] if sd else None,
                "median_abs": float(np.median(np.abs(d))),
                "within_tolerance": float(np.mean([_within(label, o, p) for o, p in pairs])),
            }
        out["measures"][label] = res
    return out


def write_outputs(rows: list[Row], summary: dict, out: Path) -> dict[str, str]:
    out = Path(out)
    table = out / "concordancia.csv"
    with open(table, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        head = ["file", "estado", "calidad"]
        for label in MEASURES:
            head += [f"{label}_programa", f"{label}_impreso", f"{label}_estado"]
        w.writerow(head + ["error"])
        for r in rows:
            line: list = [r.file, r.status, r.qc_label or ""]
            for label in MEASURES:
                o, p = r.ours.get(label), r.printed.get(label)
                line += [
                    "" if o is None else round(o, 1),
                    "" if p is None else round(p, 1),
                    r.measure_status.get(label) or "",
                ]
            w.writerow(line + [r.error or ""])
    (out / "concordancia.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = out / "concordancia.md"
    md.write_text(summary_markdown(summary), encoding="utf-8")
    plot = out / "bland_altman.png"
    bland_altman_plot(rows, plot)
    return {"table": str(table), "summary": str(md), "plot": str(plot)}


def summary_markdown(s: dict) -> str:
    lines = [
        f"# Concordancia con lo impreso ({s['processed']} de {s['photos']} fotos procesadas)",
        "",
        "| medida | n (`ok`) | sesgo | DE | límites de acuerdo 95 % | \\|dif.\\| mediana | dentro de tolerancia |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, m in s["measures"].items():
        a, o = m["all"], m["ok"]
        if not a["n"]:
            lines.append(f"| {label} | 0 de {a['printed']} impresos | | | | | |")
            continue
        loa = f"{a['loa'][0]:+.0f} a {a['loa'][1]:+.0f}" if a["loa"] else "—"
        sd = f"{a['sd']:.1f}" if a["sd"] is not None else "—"
        ok_part = f" ({o['n']})" if label != "FC" else ""
        tol = m["tolerance"] if label == "FC" else f"{m['tolerance']:.0f} {m['unit']}"
        lines.append(
            f"| {label} ({m['unit']}) | {a['n']}{ok_part} | {a['bias']:+.1f} | {sd} | {loa} | "
            f"{a['median_abs']:.1f} | {100 * a['within_tolerance']:.0f} % "
            f"(`ok`: {100 * o.get('within_tolerance', 0):.0f} %; tol. {tol}) |"
        )
    if s["not_processed"]:
        lines += [
            "",
            "No procesadas: " + ", ".join(f"{k} ({v})" for k, v in s["not_processed"].items()),
        ]
    lines += ["", "Orientativo; no es validación clínica."]
    return "\n".join(lines) + "\n"


def bland_altman_plot(rows: list[Row], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (label, (_k, _p, unit, _t)) in zip(axes.flat, MEASURES.items(), strict=True):
        pts = _pairs(rows, label, only_ok=False)
        ax.set_title(f"{label} ({unit})")
        ax.set_xlabel("media (programa, impreso)")
        ax.set_ylabel("programa − impreso")
        if not pts:
            ax.text(0.5, 0.5, "sin datos", ha="center", transform=ax.transAxes)
            continue
        mean = np.array([(o + p) / 2 for o, p, _ in pts])
        diff = np.array([_diff(label, o, p) for o, p, _ in pts])
        ok = np.array([s in (None, "ok") for _o, _p, s in pts])
        ax.scatter(mean[ok], diff[ok], s=18, label="ok")
        if (~ok).any():
            ax.scatter(mean[~ok], diff[~ok], s=18, marker="x", label="dudoso")
        ax.axhline(diff.mean(), lw=1)
        if len(diff) > 1:
            sd = diff.std(ddof=1)
            for k in (-1.96, 1.96):
                ax.axhline(diff.mean() + k * sd, lw=0.8, ls="--")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
