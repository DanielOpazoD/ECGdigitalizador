import argparse
import json
from pathlib import Path

from ecg_photo.contracts import load_manifest, validate_revision_dir
from ecg_photo.export import (
    ExportNotAllowed,
    export_csv,
    export_json,
    export_wfdb,
)
from ecg_photo.fixtures import write_fixture_revision
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
