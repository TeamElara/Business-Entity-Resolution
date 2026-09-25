"""Phase 11 France format rules, boundaries, and non-France regressions."""

import polars as pl
import pytest

from src.normalize import normalize_df


def row(name: str, address: str, country: str = "France") -> dict:
    return normalize_df(pl.DataFrame({
        "business_name": [name], "business_address": [address],
        "country": [country],
    })).row(0, named=True)


@pytest.mark.parametrize(("short", "long"), [
    ("R.", "rue"), ("BD", "boulevard"), ("AV", "avenue"),
    ("PL.", "place"), ("CH", "chemin"), ("IMP.", "impasse"),
])
def test_french_street_abbreviations(short, long):
    result = row("École SARL", f"017 {short} des Hêtres, Lille")
    assert result["addr_norm"].startswith(f"017 {long} des hetres")
    assert result["house_no"] == "017"
    assert result["postcode"] is None
    assert result["name_raw"] == "École SARL"
    assert result["name_norm"] == "ecole sarl"
    assert result["name_core"] == "ecole"
    assert result["name_latin"] == "ecole"


def test_french_short_tokens_are_contextual_and_country_guarded():
    french = row("R & CH Services", "135 RUE Coty - CH Dron, St-Nazaire")
    assert "ch dron" in french["addr_norm"]
    assert french["city"] == "saint nazaire"
    assert "rue coty" in french["addr_norm"]
    assert row("Acme", "62 RUE du 110e R. I., Bordeaux")["addr_norm"].count("rue") == 1
    us = row("École SARL", "017 BD des Hêtres, St Paul", "US")
    assert us["addr_norm"] == "017 bd des hêtres street paul"
    assert us["name_norm"] == "école sarl"
    assert row("ACME", "017 CH des Hêtres, St Paul", "Unseen")["addr_norm"] == "017 ch des hêtres street paul"


@pytest.mark.parametrize("address", [
    "N° 20 R. d'Alzon, Bordeaux", "# 403 R Colbert, Tourcoing",
    "29B R. Petite Biesse, Nantes", "R du Petit Barail, Bordeaux",
])
def test_french_rue_with_real_house_prefix_variants(address):
    assert " rue " in f" {row('École', address)['addr_norm']} "


@pytest.mark.parametrize("address", [
    "38 - CH. de la Mole, Bordeaux", "67ter Ch. des Regniers, Calais",
    "No 5 Ch. de Berrien, Saint-Nazaire",
])
def test_french_chemin_with_real_house_prefix_variants(address):
    assert " chemin " in f" {row('École', address)['addr_norm']} "


@pytest.mark.parametrize("form", ["SARL", "SAS", "SASU", "SA", "EURL", "SCI", "SNC"])
def test_french_legal_form_at_either_end(form):
    prefix = row(f"{form} École", "9 Rue des Hêtres, Lille")
    suffix = row(f"École {form}", "9 Rue des Hêtres, Lille")
    assert prefix["name_core"] == suffix["name_core"] == "ecole"
    assert prefix["legal_suffix"] == suffix["legal_suffix"] == form.lower()
    assert prefix["name_norm"] == f"{form.lower()} ecole"
    assert row(f"{form} École", "9 Main Street, Boston", "US")["name_core"] == f"{form.lower()} école"


def test_dotted_french_forms_and_suffix_only_guard():
    assert row("S.A.R.L. École", "Paris")["legal_suffix"] == "sarl"
    assert row("S.A.R.L. École", "Paris")["name_core"] == "ecole"
    assert row("École S.A.S.U.", "Paris")["legal_suffix"] == "sasu"
    only = row("SCI", "Paris")
    assert only["name_core"] == "sci" and only["legal_suffix"] is None


def test_french_accent_and_ligature_folding_keeps_raw_text():
    result = row("Cœur École SARL", "7 Rue du Cœur, Mérignac")
    assert result["name_raw"] == "Cœur École SARL"
    assert result["address_raw"] == "7 Rue du Cœur, Mérignac"
    assert result["name_core"] == "coeur ecole"
    assert result["addr_norm"] == "7 rue du coeur merignac"
    assert result["city"] == "merignac"


def test_cedex_postcode_city_and_leading_zero_house():
    compact = row("École", "042 R. de la Paix, 59046Lille Cedex")
    assert compact["postcode"] == "59046"
    assert compact["city"] == "lille"
    assert compact["house_no"] == "042"
    assert compact["addr_norm"].startswith("042 rue de la paix")
    spaced = row("École", "12 Rue de Paris, 59000 LILLE CEDEX 01")
    assert spaced["postcode"] == "59000"
    assert spaced["city"] == "lille"
    assert row("École", "12 Rue de Paris, code postal 59000, Lille")["postcode"] == "59000"


def test_french_postcode_false_positives_and_other_countries():
    assert row("École", "00109 BOULEVARD DES BELGES, NANTES")["postcode"] is None
    assert row("École", "CS 41195, BORDEAUX CEDEX")["postcode"] is None
    assert row("École", "135 Rue Coty - CH Dron, Tourcoing")["postcode"] is None
    assert row("ACME", "Main St, Atlanta, GA 30303", "US")["postcode"] == "30303"
    assert row("ACME", "H NO 23, PIN 411001, PUNE", "India")["postcode"] == "411001"
    assert row("ACME", "Postcode AB12 3CD, Central", "Unseen")["postcode"] == "AB12 3CD"
