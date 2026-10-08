"""Audit the AI research agent: every tariff number the model uses must appear, verbatim, in the
official utility PDF it is attributed to (PDFs are pinned in data/tariffs/sources/SHA256SUMS.txt)."""

import hashlib
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "tariffs", "sources")
pypdf = pytest.importorskip("pypdf")

DOCS = {
    "ngrid_summary": "ngrid_ma_summary_delivery_rates_MDPU_1-26-I_eff_2026-10-01.pdf",
    "ngrid_g3": "ngrid_ma_tariff_G-3_MDPU_1591.pdf",
    "es_summary": "eversource_ma_summary_delivery_rates_MDPU_1-26-F_eff_2026-09-01.pdf",
    "es_g2_boston": "eversource_ma_tariff_G-2_BOST_MDPU_15I.pdf",
}


def text(doc):
    path = os.path.join(SRC, DOCS[doc])
    if not os.path.exists(path):
        pytest.skip("source PDFs not present")
    return re.sub(r"\s+", " ", " ".join(p.extract_text() or "" for p in pypdf.PdfReader(path).pages))


def tariffs():
    with open(os.path.join(ROOT, "site", "data", "tariffs.json")) as f:
        return {t["id"]: t for t in json.load(f)["tariffs"]}


def test_national_grid_g3_numbers_are_in_the_rate_summary():
    t, s = tariffs()["ngrid-g3"], text("ngrid_summary")
    assert f"${t['demand'][0]['usd_per_kw']:.2f}" in s
    assert f"${t['customer_charge']:,.2f}" in s
    assert f"{t['energy']['peak_cents'] / 100:.5f}" in s
    assert f"{t['energy']['offpeak_cents'] / 100:.5f}" in s


def test_national_grid_g3_peak_window_is_in_the_tariff():
    s = text("ngrid_g3")
    assert "8:00 a.m." in s and "9:00 p.m." in s
    w = tariffs()["ngrid-g3"]["demand"][0]["peak"][0]
    assert (w["start"], w["end"], w["weekdays"]) == (8, 21, [1, 2, 3, 4, 5])


def test_eversource_g2_boston_numbers_are_in_the_rate_summary():
    t, s = tariffs()["es-g2-boston"], text("es_summary")
    assert "Demand (kW) $20.21 $20.21 $13.74 $33.95" in s
    assert t["demand"][0]["usd_per_kw"] == pytest.approx(20.21 + 13.74)
    for cap, usd in t["customer_charge_tiers"]:
        assert f"${usd:.2f}" in s
    assert f"{t['energy']['peak_cents'] / 100:.5f}" in s


def test_eversource_g2_boston_billing_rule_is_in_the_tariff():
    s = text("es_g2_boston")
    assert "reduced by 55 percent" in s
    assert "June through September, the peak period shall be the hours between 9 A.M. and 6 P.M. weekdays" in s
    assert "October through May, the peak period shall be the hours between 8 A.M. and 9 P.M. weekdays" in s
    d = tariffs()["es-g2-boston"]["demand"][0]
    assert d["offpeak_factor"] == pytest.approx(1 - 0.55)


def test_source_pdfs_match_pinned_hashes():
    sums = os.path.join(SRC, "SHA256SUMS.txt")
    if not os.path.exists(sums):
        pytest.skip("no hash manifest")
    pinned = {}
    for line in open(sums):
        parts = line.split()
        if len(parts) >= 2:
            pinned[parts[-1].lstrip("*").split("/")[-1]] = parts[0]
    for name in DOCS.values():
        h = hashlib.sha256(open(os.path.join(SRC, name), "rb").read()).hexdigest()
        assert pinned.get(name) == h, name
