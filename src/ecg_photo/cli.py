import argparse
import json
import sys
import uuid
from pathlib import Path

from ecg_photo.contracts import (
    CalibrationEvidence,
    dump_json,
    load_manifest,
    validate_revision_dir,
)
from ecg_photo.export import (
    ExportNotAllowed,
    export_csv,
    export_json,
    export_wfdb,
)
from ecg_photo.fixtures import write_fixture_revision
from ecg_photo.ingest import IngestRejected, estimate_page_grids, ingest, page_upsample
from ecg_photo.qc import qc_json, run_qc
from ecg_photo.render import (
    PaperSpec,
    RenderNotAllowed,
    render_segment_pdf,
    render_segment_png,
)

KINDS = ["calibrated", "gap", "gain_unknown", "time_unknown", "tail_2503"]


def _cmd_fixture(args: argparse.Namespace) -> int:
    write_fixture_revision(Path(args.out), args.kind, fs_hz=args.fs)
    print(f"fixture {args.kind} written to {args.out}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    manifest = load_manifest(root / "manifest.json")
    problems = validate_revision_dir(root, manifest, strict=not args.no_strict)
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


def _cmd_qc(args: argparse.Namespace) -> int:
    report = run_qc(
        Path(args.run_dir), printed_rr_ms=args.printed_rr_ms, printed_hr_bpm=args.printed_hr
    )
    text = qc_json(report)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _do_export(root: Path, out_dir: Path, dpi: float, speed: float, gain: float) -> int:
    manifest = load_manifest(root / "manifest.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    export_json(manifest, out_dir / "manifest.json")
    produced: list[str] = []
    skipped: list[dict[str, str]] = []
    produced.append("manifest.json")
    paper = PaperSpec(speed_mm_s=speed, gain_mm_mV=gain, dpi=dpi)

    for seg in manifest.segments:
        try:
            export_csv(root, seg, out_dir / f"{seg.segment_id}.csv")
            produced.append(f"{seg.segment_id}.csv")
        except ExportNotAllowed as e:
            skipped.append(
                {"segment_id": seg.segment_id, "artifact": "csv", "reason_code": str(e.reason)}
            )
        for artifact, fn in (
            ("png", render_segment_png),
            ("pdf", render_segment_pdf),
        ):
            try:
                fn(root, seg, out_dir / f"{seg.segment_id}.{artifact}", paper)
                produced.append(f"{seg.segment_id}.{artifact}")
            except RenderNotAllowed as e:
                skipped.append(
                    {
                        "segment_id": seg.segment_id,
                        "artifact": artifact,
                        "reason_code": str(e.reason),
                    }
                )
        try:
            export_wfdb(root, manifest, [seg.segment_id], out_dir, seg.segment_id)
            produced.append(f"{seg.segment_id}.hea/.dat (wfdb)")
        except ExportNotAllowed as e:
            skipped.append(
                {"segment_id": seg.segment_id, "artifact": "wfdb", "reason_code": str(e.reason)}
            )

    report = {"produced": produced, "skipped": skipped}
    (out_dir / "export_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for p in produced:
        print(f"produced  {p}")
    for s in skipped:
        print(f"skipped   {s['segment_id']} {s['artifact']} {s['reason_code']}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    return _do_export(Path(args.dir), Path(args.out), args.dpi, args.speed, args.gain)


def _cmd_serve(args: argparse.Namespace) -> int:
    import ipaddress

    from ecg_photo.api import create_app
    from ecg_photo.ingest import load_supported_inputs
    from ecg_photo.store import Store, StoreLocked, load_execution_limits
    from ecg_photo.worker import Worker, default_engines, resume_after_restart

    host = args.host
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host in ("localhost",)
    if not loopback and not args.allow_non_loopback:
        print(
            json.dumps(
                {"error": "refusing non-loopback host; pass --allow-non-loopback to override (F9)"}
            )
        )
        return 2
    store = Store(
        Path(args.store),
        load_supported_inputs(),
        load_execution_limits(),
    )
    try:
        store.acquire_process_lock()
    except StoreLocked as e:
        print(json.dumps({"error": str(e), "code": e.code}))
        return 2
    worker = Worker(store, default_engines())
    worker.start()
    print(json.dumps({"recovery": resume_after_restart(store, worker)}))
    app = create_app(store, worker)
    import uvicorn

    uvicorn.run(app, host=host, port=args.port)
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    from ecg_photo.batch import BatchError, list_inputs, run_batch
    from ecg_photo.ingest import load_supported_inputs
    from ecg_photo.store import Store, StoreLocked, StudyPatch, load_execution_limits
    from ecg_photo.worker import default_engines

    if (args.speed is not None or args.gain is not None) and not (args.author and args.reason):
        print(json.dumps({"error": "--speed/--gain are manual evidence: --author and --reason"}))
        return 2
    options = StudyPatch(
        **{
            k: v
            for k, v in {
                "engine": args.engine,
                "engine_duration_s": args.duration,
                "time_source": args.time_source,
                "speed_mm_s": args.speed,
                "gain_mm_mV": args.gain,
                "fs_hz": args.fs,
                "page_id": args.page,
            }.items()
            if v is not None
        }
    )
    store = Store(Path(args.store), load_supported_inputs(), load_execution_limits())
    try:
        store.acquire_process_lock()
    except StoreLocked as e:
        print(json.dumps({"error": str(e), "code": e.code}))
        return 2
    try:
        recovery = store.recover()
        report = run_batch(
            store,
            default_engines(args.engines_config),
            list_inputs(Path(args.input_dir)),
            options,
            Path(args.report),
            author=args.author or "batch",
            reason=args.reason or "batch",
            resume=args.resume,
        )
    except BatchError as e:
        print(json.dumps({"error": str(e)}))
        return 2
    finally:
        store.release_process_lock()
    print(
        json.dumps(
            {
                "report": str(args.report),
                "summary": report["summary"],
                # runs from an earlier `serve` left waiting; `serve` resumes them
                "left_queued": recovery.as_dict()["requeue"],
            }
        )
    )
    return 0 if set(report["summary"]) <= {"published"} else 1


def _cmd_run_demo(args: argparse.Namespace) -> int:
    out = Path(args.out)
    root = out / "revision"
    write_fixture_revision(root, "calibrated")
    return _do_export(root, out / "export", 300.0, 25.0, 10.0)


def _cmd_ingest(args: argparse.Namespace) -> int:
    out = Path(args.out)
    try:
        manifest = ingest(Path(args.file), out, render_dpi=args.render_dpi)
    except IngestRejected as e:
        print(json.dumps({"rejected": str(e.reason_code)}, ensure_ascii=False))
        return 2
    pages_summary: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "study_id": manifest.study_id,
        "pages": pages_summary,
    }
    manifest, grids = estimate_page_grids(out, manifest)
    for pg, grid in zip(manifest.pages, grids, strict=True):
        pages_summary.append(
            {
                "page_id": pg.page_id,
                "extraction": str(pg.extraction),
                "size_px": [pg.width_px, pg.height_px],
                "exif_orientation": pg.exif_orientation,
                # low-resolution photos are upsampled at ingest; the grid is
                # measured on the page raster, so px/mm are page pixels
                "upsample": page_upsample(manifest, pg.page_id),
                "px_per_mm_x": grid.px_per_mm_x,
                "px_per_mm_y": grid.px_per_mm_y,
            }
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    manifest = load_manifest(root / "manifest.json")
    cal_path = root / "calibration.json"
    entries: list[dict] = json.loads(cal_path.read_text()) if cal_path.exists() else []
    for quantity, value, unit in (
        ("speed_mm_s", args.speed, "mm/s"),
        ("gain_mm_mV", args.gain, "mm/mV"),
    ):
        if value is None:
            continue
        ev = CalibrationEvidence(
            evidence_id=f"manual-{uuid.uuid4().hex[:12]}",
            kind="manual",
            quantity=quantity,  # type: ignore[arg-type]
            value=value,
            unit=unit,
            region=None,
            frame_id=args.page,
            method="cli-calibrate",
            limitations="manual operator value; not verified against raster",
            author=args.author,
            reason=args.reason,
        )
        entries.append(ev.model_dump(mode="json"))
    cal_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    old_rev = manifest.revision
    (root / f"manifest.rev{old_rev}.json").write_text(
        (root / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    manifest = manifest.model_copy(update={"revision": old_rev + 1})
    dump_json(manifest, root / "manifest.json")
    print(json.dumps({"revision": manifest.revision, "evidence_added": len(entries)}, indent=2))
    return 0


def _cmd_digitize(args: argparse.Namespace) -> int:
    from ecg_photo.digitizers.base import Digitizer

    digitizer: Digitizer
    if args.engine == "ahus":
        missing = [
            a for a in ("ahus_root", "ahus_python", "ahus_config") if getattr(args, a) is None
        ]
        if missing:
            print(json.dumps({"error": f"missing args: {missing}"}))
            return 2
        from ecg_photo.digitizers.ahus import AhusDigitizer

        digitizer = AhusDigitizer(
            ahus_root=args.ahus_root.resolve(),
            python_exe=args.ahus_python.absolute(),
            base_config=args.ahus_config.resolve(),
            duration_s=args.duration,
            device="cpu",
        )
    else:
        missing = [a for a in ("digitiser_root", "digitiser_python") if getattr(args, a) is None]
        if missing:
            print(json.dumps({"error": f"missing args: {missing}"}))
            return 2
        from ecg_photo.digitizers.ecg_digitiser import EcgDigitiserDigitizer

        digitizer = EcgDigitiserDigitizer(
            root=args.digitiser_root.resolve(),
            python_exe=args.digitiser_python.absolute(),
            model_dir=args.digitiser_model,
        )
    from ecg_photo.pipeline import digitize_page

    try:
        paths = digitize_page(
            Path(args.dir),
            args.page,
            digitizer,
            engine_duration_s=args.duration,
            runs_root=args.runs_root,
        )
    except (ValueError, RuntimeError) as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    report = json.loads((paths.run_dir / "run_report.json").read_text())
    print(
        json.dumps(
            {
                "run_dir": str(paths.run_dir),
                "leads_written": report["leads_written"],
                "skipped": report["skipped_leads"],
                "layout": report["layout_detected"],
            }
        )
    )
    return 0


def _cmd_confirm_scale(args: argparse.Namespace) -> int:
    from ecg_photo.pipeline import confirm_scale

    try:
        paths = confirm_scale(
            Path(args.dir),
            gain_mm_mV=args.gain,
            author=args.author,
            reason=args.reason,
            speed_mm_s=args.speed,
            fs_hz=args.fs,
            time_source=args.time_source,
        )
    except (ValueError, RuntimeError) as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    print(json.dumps({"run_dir": str(paths.run_dir)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ecg-photo")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fixture")
    f.add_argument("--kind", choices=KINDS, required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--fs", type=float, default=500.0)
    f.set_defaults(func=_cmd_fixture)

    v = sub.add_parser("validate")
    v.add_argument("dir")
    v.add_argument("--no-strict", action="store_true")
    v.set_defaults(func=_cmd_validate)

    q = sub.add_parser("qc", help="informe de calidad de una corrida confirmada (sin verdad)")
    q.add_argument("run_dir")
    q.add_argument("--out", default=None, help="escribe el informe JSON en este archivo")
    q.add_argument(
        "--printed-rr-ms", type=float, default=None, help="RR impreso por el equipo (ms)"
    )
    q.add_argument("--printed-hr", type=float, default=None, help="FC impresa por el equipo (lpm)")
    q.set_defaults(func=_cmd_qc)

    e = sub.add_parser("export")
    e.add_argument("dir")
    e.add_argument("--out", required=True)
    e.add_argument("--dpi", type=float, default=300.0)
    e.add_argument("--speed", type=float, default=25.0)
    e.add_argument("--gain", type=float, default=10.0)
    e.set_defaults(func=_cmd_export)

    d = sub.add_parser("run-demo")
    d.add_argument("--out", required=True)
    d.set_defaults(func=_cmd_run_demo)

    i = sub.add_parser("ingest")
    i.add_argument("file")
    i.add_argument("--out", required=True)
    i.add_argument("--render-dpi", type=float, default=300.0)
    i.set_defaults(func=_cmd_ingest)

    c = sub.add_parser("calibrate")
    c.add_argument("dir")
    c.add_argument("--page", required=True)
    c.add_argument("--speed", type=float, default=None)
    c.add_argument("--gain", type=float, default=None)
    c.add_argument("--author", required=True)
    c.add_argument("--reason", required=True)
    c.set_defaults(func=_cmd_calibrate)

    dg = sub.add_parser("digitize")
    dg.add_argument("dir")
    dg.add_argument("--page", required=True)
    dg.add_argument("--engine", choices=["ahus", "ecg-digitiser"], required=True)
    dg.add_argument("--duration", type=float, required=True)
    dg.add_argument("--ahus-root", type=Path)
    dg.add_argument("--ahus-python", type=Path)
    dg.add_argument("--ahus-config", type=Path)
    dg.add_argument("--digitiser-root", type=Path)
    dg.add_argument("--digitiser-python", type=Path)
    dg.add_argument("--digitiser-model", type=Path, default=Path("models/M3"))
    dg.add_argument("--runs-root", type=Path, default=None)
    dg.set_defaults(func=_cmd_digitize)

    cs = sub.add_parser("confirm-scale")
    cs.add_argument("dir")
    cs.add_argument("--speed", type=float, default=None)
    cs.add_argument("--gain", type=float, required=True)
    cs.add_argument("--author", required=True)
    cs.add_argument("--reason", required=True)
    cs.add_argument("--fs", type=float, default=None)
    cs.add_argument("--time-source", choices=["evidence", "engine"], default="evidence")
    cs.set_defaults(func=_cmd_confirm_scale)

    sv = sub.add_parser("serve")
    sv.add_argument("--store", required=True, help="store directory")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="permitir bind fuera de loopback (F9; desactivado por defecto)",
    )
    sv.set_defaults(func=_cmd_serve)

    b = sub.add_parser("batch", help="procesa todos los archivos de un directorio (T48/T49)")
    b.add_argument("input_dir")
    b.add_argument("--store", required=True)
    b.add_argument("--report", required=True, help="informe JSON (se reescribe tras cada archivo)")
    b.add_argument("--resume", action="store_true", help="continuar un informe existente")
    b.add_argument("--engine", required=True)
    b.add_argument("--duration", type=float, default=None)
    b.add_argument("--time-source", choices=["evidence", "engine"], default=None)
    b.add_argument("--speed", type=float, default=None)
    b.add_argument("--gain", type=float, default=None)
    b.add_argument("--fs", type=float, default=None)
    b.add_argument("--page", default=None)
    b.add_argument("--author", default=None)
    b.add_argument("--reason", default=None)
    b.add_argument("--engines-config", type=Path, default=None)
    b.set_defaults(func=_cmd_batch)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
