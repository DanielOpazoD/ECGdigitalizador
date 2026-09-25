import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from f6b_kaggle_bench import LEADS, short_error, with_missing_leads


def test_short_error_keeps_the_exception_at_the_end() -> None:
    # real case (F6-b, ECG-Digitiser, 1561472702-0003): stderr starts with tqdm
    # progress and the traceback is at the end; the old [:400] kept only progress
    progress = "".join(f" {i}/27 [00:{i:02d}<04:07, 12.36s/it]\n" for i in range(200))
    e = RuntimeError(progress + "Traceback ...\nValueError: the real cause")
    msg = short_error(e)
    assert msg.startswith("RuntimeError: ")
    assert msg.endswith("ValueError: the real cause")
    assert len(msg) < 1400
    assert short_error(ValueError("short")) == "ValueError: short"


def test_missing_leads_become_failure_cases() -> None:
    # real case (F6-b, ECG-Digitiser, 1512936796-0005): 11 leads returned, no III
    cases = [{"lead": ld, "status": "ok"} for ld in LEADS if ld != "III"]
    out = with_missing_leads(cases)
    assert len(out) == 12
    assert [c for c in out if c["status"] == "lead_missing"] == [
        {"lead": "III", "status": "lead_missing"}
    ]
    per_image = [{"status": "engine_error", "error": "x"}]
    assert with_missing_leads(per_image) == per_image
