"""Opt-in house-number and address-noise fields for v4 evaluation."""

import polars as pl

from src.normalize import normalize_df


def row(address: str, country: str = "France") -> dict:
    return normalize_df(pl.DataFrame({
        "entity_id": ["x"], "country": [country],
        "business_name": ["Example"], "business_address": [address],
    }), transliteration_map={}).row(0, named=True)


def test_leading_zero_and_french_suffix_variants():
    assert row("001511 Rue du Port, Calais")["house_no2"] == "1511"
    assert row("1511 Rue du Port, Calais")["house_no2"] == "1511"
    assert row("12 bis Rue du Port, Calais")["house_no2"] == "12bis"
    assert row("12B Rue du Port, Calais")["house_no2"] == "12bis"
    assert row("12 ter Rue du Port, Calais")["house_no2"] == "12ter"
    assert row("1 d Boulevard des Crêtes, Pessac")["house_no2"] == "1d"
    assert row("1D Boulevard des Crêtes, Pessac")["house_no2"] == "1d"


def test_different_buildings_stay_different():
    assert row("100 Rue de Paris, Calais")["house_no2"] == "100"
    assert row("10 Rue de Paris, Calais")["house_no2"] == "10"
    assert row("12 Rue de Paris, Calais")["house_no2"] == "12"
    assert row("12 bis Rue de Paris, Calais")["house_no2"] == "12bis"


def test_street_number_wins_over_unit_and_multiple_numbers():
    fr = row("Apt 6, No. 551 R. Anatole France, Dunkerque")
    assert fr["house_no2"] == "551"
    assert "apartment 6" not in fr["addr_norm2"]
    assert "551" in fr["addr_norm2"]
    assert row("6 Impasse Saint Louis 215 Rue des Cinq Voies, Lille")["house_no2"] == "6"
    us = row("Unit 4, 100 Main St, Boston, MA", "US")
    assert us["house_no2"] == "100"
    assert "unit 4" not in us["addr_norm2"]
    assert row("DC, WASHINGTON, 2117 F STREET", "US")["house_no2"] == "2117"
    assert row("90 N S RD, KOLKATA", "India")["house_no2"] == "90"


def test_legacy_fields_are_untouched():
    fr = row("No. 001511 Rue du Port, Calais")
    assert fr["house_no"] == "00151"
    assert "001511" in fr["addr_norm"]
    india = row("H.NO: 2-6-971/6/AZ-B5, HANAMKONDA, Telangana", "India")
    assert india["house_no2"] == "2-6-971/6/az-b5"
    assert india["addr_norm2"] == india["addr_norm"]
    assert row("SY NO 26/1.16 HOSTEL RD, SHOP NO. 5, BANGALORE", "India")["house_no2"] == "26/1.16"


def test_missing_address_stays_missing():
    out = row(None)
    assert out["house_no2"] is None
    assert out["addr_norm2"] == ""
