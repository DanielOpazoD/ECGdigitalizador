from ecg_photo.report import measurement_rows, reason_es


def test_measurement_rows_status_and_spanish_reasons() -> None:
    iv = {
        "rr_ms": 800.0,
        "hr_bpm": 75.0,
        "pr_ms": None,
        "qrs_ms": 96.0,
        "qt_ms": 380.0,
        "qtc_bazett_ms": 424.9,
        "qtc_fridericia_ms": 409.3,
        "spread_ms": {"pr_ms": None, "qrs_ms": 6.0, "qt_ms": 55.0},
        "status": {"pr_ms": "unavailable", "qrs_ms": "ok", "qt_ms": "doubtful"},
        "n_leads": {"pr_ms": 0, "qrs_ms": 12, "qt_ms": 9},
        "reasons": {"pr_ms": "not found in any lead", "qt_ms": "leads disagree (IQR 55 ms > 40)"},
    }
    rows = {name: (value, note) for name, value, note in measurement_rows(iv)}
    assert rows["Frecuencia"] == ("75 /min", "RR 800 ms")
    assert rows["PR"] == ("—", "no hallado en ninguna derivación")
    assert rows["QRS"] == ("96 ms", "12 derivaciones, dispersión 6 ms")
    assert rows["QT"][1] == "dudoso: derivaciones discordantes (IQR 55 ms > 40)"
    # QTc inherits the doubt of QT
    assert rows["QTc"][0] == "425 ms" and rows["QTc"][1].startswith("dudoso")


def test_missing_or_failed_intervals() -> None:
    assert measurement_rows(None) == [("Intervalos", "—", "no calculados")]
    assert "boom" in measurement_rows({"error": "ValueError: boom"})[0][2]
    assert reason_es("found in 2 lead(s) only") == "hallado sólo en 2 derivación(es)"
    assert reason_es("something new") == "something new"
