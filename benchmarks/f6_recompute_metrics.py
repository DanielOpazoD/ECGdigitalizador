"""Recompute F6-a per-lead metrics from existing runs/f6 run dirs (no engine
re-run). Rewrites `cases` + `aggregates` in runs/f6/f6_results.json.

Same scope caveat as the main bench: real signals, synthetic images —
NOT real photographs, NOT clinical validation.
"""

import json
import sys
from pathlib import Path

import numpy as np
import wfdb  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f6_ptbxl_bench import LEADS, aggregate, rhythm_lead_of
from metrics_common import segment_metrics

from ecg_photo.contracts import load_manifest


def main() -> int:
    work = Path("runs/f6").resolve()
    res = json.loads((work / "f6_results.json").read_text())
    records = sorted(int(r[:5]) for r in res["ptbxl"]["records"])

    cases: dict[str, list[dict]] = {e: [] for e in res["engines"]}
    for n in records:
        img_dir = work / f"imgkit_{n:05d}"
        gt = json.loads(min(img_dir.glob("*.json")).read_text())
        rec = wfdb.rdrecord(str(work / "ptbxl" / f"{n:05d}_hr"))
        truth, sig_names, truth_fs = (
            rec.p_signal,
            [s.upper() for s in rec.sig_name],
            float(rec.fs),
        )
        rhythm = rhythm_lead_of(gt, int(truth.shape[0]))

        # confirmed runs for this record: run dirs with signal_path segments
        rev_dir = work / f"rec{n:05d}" / "rev"
        runs = sorted(p.parent for p in (rev_dir / "runs").glob("run-*/manifest.json"))
        for engine in res["engines"]:
            # confirmed run dirs carry no run_report.json; identify the engine
            # by working_fs_hz (ahus emits 1000 Hz canonical, digitiser 500 Hz)
            want_fs = {"ahus": 1000.0, "ecg-digitiser": 500.0}[engine]
            match = None
            match_dir = None
            for rd in runs:
                m = load_manifest(rd / "manifest.json")
                if any(s.signal_path for s in m.segments) and any(
                    s.working_fs_hz == want_fs for s in m.segments
                ):
                    match = m
                    match_dir = rd
            if match is None:
                for lead in LEADS:
                    cases[engine].append({"record": n, "lead": lead, "status": "engine_error"})
                continue
            for seg in match.segments:
                lead = seg.lead_label
                if lead not in LEADS or seg.signal_path is None:
                    cases[engine].append(
                        {
                            "record": n,
                            "lead": lead,
                            "status": "no_signal",
                            "segment_id": seg.segment_id,
                        }
                    )
                    continue
                sig = np.load(match_dir / seg.signal_path, allow_pickle=False)
                observed = np.isfinite(sig)
                if seg.observed_mask_path and (match_dir / seg.observed_mask_path).exists():
                    observed = np.load(match_dir / seg.observed_mask_path, allow_pickle=False)
                tlead = truth[:, sig_names.index(lead.upper())]
                m_met = segment_metrics(
                    tlead,
                    sig,
                    observed,
                    est_fs=float(seg.working_fs_hz or truth_fs),
                    truth_fs=truth_fs,
                    observed_duration_s=seg.observed_duration_s,
                    expected_window_s=10.0 if lead == rhythm else 2.5,
                )
                cases[engine].append(
                    {"record": n, "lead": lead, "segment_id": seg.segment_id, **m_met}
                )

    res["cases"] = cases
    res["aggregates"] = {e: aggregate(c) for e, c in cases.items()}
    (work / "f6_results.json").write_text(json.dumps(res, indent=2))
    for e, a in res["aggregates"].items():
        print(
            f"{e}: n_ok {a['n_ok']}/{a['n_leads']} r {a['median_r_at_lag']} "
            f"rmse {a['median_rmse_mV']} amp {a['median_amp_ratio']} "
            f"dur_err {a['median_duration_err_pct']} r>=0.9 {a['frac_r_ge_0.9']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
