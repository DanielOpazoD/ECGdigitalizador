"""One-page PDF report of a confirmed run (`report.pdf`): the redrawn ECG,
heart rate and intervals with their status, the quality label, how the time
and voltage scales were obtained, and the limits of use. It only presents
what qc.json, intervals.json and the manifest already hold; nothing is
recomputed or inferred here.
"""

import re
import tempfile
from pathlib import Path

from ecg_photo.contracts import Manifest, ScaleStatus, load_manifest
from ecg_photo.intervals import read_intervals_report
from ecg_photo.qc import read_qc_report

REPORT_FILENAME = "report.pdf"
DISCLAIMER = (
    "Digitalización automática de una imagen de ECG. Orientativo: no es validación clínica ni "
    "sustituye la lectura del trazado original. Los valores marcados «dudoso» o «—» no deben "
    "usarse sin revisar el original."
)
QC_TEXT = {
    "good": "buena",
    "acceptable": "aceptable",
    "insufficient": "insuficiente: revisar el original",
}


REASONS_ES = (
    (
        re.compile(r"leads disagree \(IQR (\d+) ms > (\d+)\)"),
        r"derivaciones discordantes (IQR \1 ms > \2)",
    ),
    (re.compile(r"found in (\d+) lead\(s\) only"), r"hallado sólo en \1 derivación(es)"),
    (re.compile(r"not found in any lead"), "no hallado en ninguna derivación"),
    (re.compile(r"digitization quality insufficient"), "calidad de digitalización insuficiente"),
    (re.compile(r"RR not measured"), "RR no medido"),
    (
        re.compile(r"differs from printed \w+ by ([+-]\d+) ms \(> (\d+)\)"),
        r"difiere del impreso en \1 ms (> \2)",
    ),
    (re.compile(r"RR (\d+) ms outside .*"), r"RR \1 ms fuera de rango"),
)


def reason_es(text: str) -> str:
    """Spanish wording of an intervals.json reason (unknown text unchanged)."""
    for pat, repl in REASONS_ES:
        text = pat.sub(repl, text)
    return text


def _ms(v: float | None) -> str:
    return "—" if v is None else f"{v:.0f} ms"


def measurement_rows(iv: dict | None) -> list[tuple[str, str, str]]:
    """(name, value, note) rows for the report table."""
    if iv is None:
        return [("Intervalos", "—", "no calculados")]
    if "error" in iv:
        return [("Intervalos", "—", f"no calculados ({iv['error']})")]
    status = iv.get("status") or {}
    reasons = iv.get("reasons") or {}
    spread = iv.get("spread_ms") or {}
    n_leads = iv.get("n_leads") or {}

    printed = iv.get("printed") or {}

    def with_printed(key: str, text: str) -> str:
        p = printed.get(key)
        if not p:
            return text
        mark = {True: "coincide", False: "NO coincide", None: "sin medida"}[p.get("agrees")]
        return f"{text}; impreso {p['printed_ms']:.0f} ms ({mark})"

    def note(key: str) -> str:
        return with_printed(key, _note(key))

    def _note(key: str) -> str:
        st = status.get(key)
        if st == "ok":
            return f"{n_leads.get(key, 0)} derivaciones, dispersión {spread.get(key) or 0:.0f} ms"
        if st == "doubtful":
            return f"dudoso: {reason_es(reasons.get(key, ''))}"
        return reason_es(reasons.get(key, "no medible"))

    hr = iv.get("hr_bpm")
    rows = [
        (
            "Frecuencia",
            "—" if hr is None else f"{hr:.0f} /min",
            "—" if iv.get("rr_ms") is None else f"RR {iv['rr_ms']:.0f} ms",
        ),
        ("PR", _ms(iv.get("pr_ms")), note("pr_ms")),
        ("QRS", _ms(iv.get("qrs_ms")), note("qrs_ms")),
        ("QT", _ms(iv.get("qt_ms")), note("qt_ms")),
    ]
    qtc_note = reason_es(
        reasons.get("qtc", "Bazett; Fridericia " + _ms(iv.get("qtc_fridericia_ms")))
    )
    if status.get("qt_ms") != "ok" and iv.get("qtc_bazett_ms") is not None:
        qtc_note = "dudoso (QT dudoso); " + qtc_note
    rows.append(("QTc", _ms(iv.get("qtc_bazett_ms")), with_printed("qtc_bazett_ms", qtc_note)))
    return rows


def scale_lines(manifest: Manifest) -> list[str]:
    """How speed, gain and the time axis of the published leads were obtained."""
    segs = [s for s in manifest.segments if s.signal_path is not None]
    if not segs:
        return ["Sin derivaciones calibradas."]
    s = segs[0]
    stages = {p.stage for seg in segs for p in seg.processing}
    evidence = "time_axis_from_image_evidence" in stages and s.effective_dt_ms is not None
    speed_how = "confirmada por el usuario" if s.speed_status == ScaleStatus.manual else "medida"
    lines = [
        f"Velocidad {s.speed_mm_s:g} mm/s ({speed_how}).",
        f"Ganancia {s.gain_mm_mV:g} mm/mV (confirmada por el usuario).",
        "Eje temporal: "
        + (
            "medido en la imagen (rejilla × velocidad)."
            if evidence
            else "duración supuesta por el motor, confirmada."
        ),
    ]
    missing = 12 - len({seg.lead_label for seg in segs})
    if missing > 0:
        lines.append(f"{missing} derivación(es) sin señal digitalizada.")
    return lines


def render_report(run_dir: Path, out_pdf: Path, *, source_name: str | None = None) -> None:
    """Write the one-page A4 landscape report of a confirmed run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    from ecg_photo.process import OVERVIEW_FILENAME, render_overview

    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir / "manifest.json")
    qc = read_qc_report(run_dir)
    iv = read_intervals_report(run_dir)
    name = source_name or manifest.source.original_filename or manifest.study_id

    overview = run_dir / OVERVIEW_FILENAME
    with tempfile.TemporaryDirectory() as tmp:
        if not overview.exists():
            overview = Path(tmp) / OVERVIEW_FILENAME
            render_overview(run_dir, overview)
        img = mpimg.imread(str(overview))

    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.text(0.04, 0.955, "ECG digitalizado desde imagen", fontsize=15, weight="bold")
    fig.text(
        0.04,
        0.925,
        f"Archivo: {name}   ·   estudio {manifest.study_id}   ·   corrida {manifest.run_id}",
        fontsize=8.5,
        color="#444",
    )
    ax = fig.add_axes((0.04, 0.36, 0.92, 0.55))
    ax.imshow(img)
    ax.set_axis_off()

    label = (qc or {}).get("label")
    flags = ", ".join((qc or {}).get("flags") or []) or "ninguna"
    y = 0.315
    fig.text(0.04, y, "Calidad de la digitalización", fontsize=10.5, weight="bold")
    fig.text(
        0.04,
        y - 0.03,
        f"{QC_TEXT.get(label, 'no evaluada') if label else 'no evaluada'}  (alertas: {flags})",
        fontsize=9,
        color="#b00020" if label == "insufficient" else "#000",
    )
    for i, line in enumerate(scale_lines(manifest)):
        fig.text(0.04, y - 0.065 - 0.027 * i, line, fontsize=8.5)

    fig.text(0.52, y, "Mediciones (entre derivaciones)", fontsize=10.5, weight="bold")
    for i, (k, v, note) in enumerate(measurement_rows(iv)):
        yy = y - 0.03 - 0.034 * i
        doubtful = note.startswith("dudoso")
        fig.text(0.52, yy, k, fontsize=9, weight="bold")
        fig.text(0.615, yy, v, fontsize=9, color="#b00020" if doubtful else "#000")
        main, _, printed = note.partition("; impreso ")
        fig.text(0.695, yy, main[:62], fontsize=7.5, color="#555")
        if printed:
            mismatch = "NO coincide" in printed
            fig.text(
                0.695,
                yy - 0.014,
                ("impreso " + printed)[:62],
                fontsize=7,
                color="#b00020" if mismatch else "#555",
            )

    fig.text(0.04, 0.035, DISCLAIMER, fontsize=7.5, color="#444", wrap=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def write_report(run_dir: Path, *, source_name: str | None = None) -> dict:
    """`report.pdf` next to a confirmed run's manifest. Never raises: the
    report is a presentation of results that already exist."""
    try:
        render_report(run_dir, Path(run_dir) / REPORT_FILENAME, source_name=source_name)
    except Exception as e:  # noqa: BLE001 - advisory document, the run stands
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    return {"ok": True}
