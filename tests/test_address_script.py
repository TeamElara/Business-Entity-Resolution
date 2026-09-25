"""Regression checks for the conservative Phase 6/7 extraction rules."""

import polars as pl

from src.normalize import normalize_df


def _one(address, country="US", name="ACME Ltd"):
    return normalize_df(
        pl.DataFrame(
            {"entity_id": ["1"], "country": [country],
             "business_name": [name], "business_address": [address]}
        )
    ).row(0, named=True)


def test_postcode_requires_postal_context():
    assert _one("2070 QUEENS BLUFF WAY, CLARKSVILLE, TN")["postcode"] is None
    assert _one("14004 HOLIDAY DRIVE, MOUNT ORAB, OH")["postcode"] is None
    assert _one("Main St, Atlanta, GA 30303")["postcode"] == "30303"
    assert _one("36234 Aspen Court, WI, City Of Independence, Fl 13887")["postcode"] is None
    assert _one("Fl. 0, Rocksprings, Texas, 255 Sd 46092")["postcode"] is None
    assert _one("Main St, Atlanta, ZIP 30303-1234")["postcode"] == "30303-1234"
    assert _one("00109 BOULEVARD DES BELGES, NANTES", "France")["postcode"] is None
    assert _one("00109, NANTES", "France")["postcode"] is None
    assert _one("195 R. DE LA MACKELLERIE, 59100, ROUBAIX", "France")["postcode"] == "59100"
    assert _one("195 RUE DE PARIS, 59000 LILLE", "France")["postcode"] == "59000"
    assert _one("CS 11235, BORDEAUX", "France")["postcode"] is None
    assert _one("H NO 23, PUNE 411001", "India")["postcode"] == "411001"
    assert _one("H NO 23, PIN 411001, PUNE", "India")["postcode"] == "411001"
    assert _one("H NO 23, 411001, PUNE", "India")["postcode"] == "411001"
    assert _one("H NO 23, PUNE", "India")["postcode"] is None
    assert _one("Pinnacle Building, Pune", "India")["postcode"] is None
    assert _one("S No_416660, Hassan Mysore Main Road, Karnataka", "India")["postcode"] is None
    assert _one("Postcode AB12 3CD, Central", "UK")["postcode"] == "AB12 3CD"
    assert _one("Main Square, 123456, Central", "Unknown")["postcode"] == "123456"


def test_city_and_house_fields():
    row = _one("2070 QUEENS BLUFF WAY, CLARKSVILLE, TN")
    assert row["city"] == "clarksville"
    assert row["house_no"] == "2070"
    row = _one("H.NO: 2-6-971/6/AZ-B5, null, HANAMKONDA, WARANGAL URBAN, Telangana", "India")
    assert row["city"] == "warangal urban"
    assert row["house_no"] is not None
    row = _one("195 R. DE LA MACKELLERIE, 59100, ROUBAIX, Nord", "France")
    assert row["city"] == "roubaix"
    assert row["house_no"] == "195"
    assert _one("153 R DES SOUPIRANTS, 62100, CALAIS, Pas-de-Calais", "France")["city"] == "calais"
    row = _one(None)
    assert row["postcode"] is None and row["city"] is None and row["house_no"] is None


def test_script_and_latin_name():
    latin = _one("Paris", "France", "École SARL")
    assert latin["script"] == "latin"
    assert latin["name_latin"] == "ecole"
    indic = _one("Pune", "India", "भारत ट्रेडर्स")
    assert indic["script"] == "devanagari"
    assert indic["name_latin"]
    assert indic["name_latin"].isascii()
    other = _one("Chennai", "India", "தமிழ் கடை")
    assert other["script"] == "other"
    assert other["name_latin"] == ""
    missing = _one(None, "India", None)
    assert missing["script"] == "latin"
    assert missing["name_latin"] == ""
