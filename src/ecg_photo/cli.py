import argparse
import json
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

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
from ecg_photo.grid import estimate_grid
from ecg_photo.ingest import IngestRejected, ingest
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
    for pg in manifest.pages:
        raster = out / pg.raster_path
        img = np.asarray(Image.open(raster))
        grid = estimate_grid(img)
        grid_path = out / "pages" / f"{pg.page_id}.grid.json"
        grid_path.write_text(
            json.dumps(
                {
                    "px_per_mm_x": grid.px_per_mm_x,
                    "px_per_mm_y": grid.px_per_mm_y,
                    "minor_period_px_x": grid.minor_period_px_x,
                    "minor_period_px_y": grid.minor_period_px_y,
                    "major_period_px_x": grid.major_period_px_x,
                    "major_period_px_y": grid.major_period_px_y,
                    "residual_rel_x": grid.residual_rel_x,
                    "residual_rel_y": grid.residual_rel_y,
                    "method": grid.method,
                    "limitations": grid.limitations,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        pg = pg.model_copy(update={"grid_estimate_path": f"pages/{pg.page_id}.grid.json"})
        manifest.pages[pg.index] = pg
        pages_summary.append(
            {
                "page_id": pg.page_id,
                "extraction": str(pg.extraction),
                "size_px": [pg.width_px, pg.height_px],
                "exif_orientation": pg.exif_orientation,
                "px_per_mm_x": grid.px_per_mm_x,
                "px_per_mm_y": grid.px_per_mm_y,
            }
        )
    dump_json(manifest, out / "manifest.json")
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
