"""Conservative address-field extraction from the supplied challenge text.

The dataset contains many five-digit house numbers but almost no postcodes.
Extract a postcode only in a postal context, leaving uncertain cases null.
All expressions work on Polars columns, so no per-row Python callbacks are used.
"""

import polars as pl

from .basic_text import clean_text, strip_accents


# Labels observed in the provided S1 inputs. They identify a final region
# component that should not be mistaken for a city. Unknown countries still
# use the generic comma-segment rules below.
REGION_LABELS = (
    "maharashtra", "delhi", "uttar pradesh", "karnataka", "tamil nadu",
    "gujarat", "west bengal", "telangana", "haryana", "kerala",
    "rajasthan", "bihar", "madhya pradesh", "andhra pradesh", "orissa",
    "odisha", "punjab", "jharkhand", "assam", "uttarakhand",
    "hauts-de-france", "nouvelle-aquitaine", "pays de la loire",
    "nord", "gironde", "loire-atlantique", "pas-de-calais",
)

PLACEHOLDERS = ("", "null", "<null>", "none", "na", "n/a", '""')

# An address segment containing these words is more likely a street/building
# description than a city. Kept deliberately short to avoid hiding real cities.
STREET_WORDS = (
    r"(?i)(?:^|\s)(?:road|rd|street|st|avenue|ave|boulevard|blvd|"
    r"lane|ln|rue|route|rte|chemin|impasse|plot|block|floor|flr|"
    r"apartment|apt|building|bldg|flat|shop|sector|highway|hwy)(?:\s|$)"
)
FRANCE_EXTRA_STREET_WORDS = r"(?i)(?:^|\s)(?:r|ch|imp|bd|pl)(?:\s|$)"

# Keep alphanumeric house-number structure such as 16-2-851/B or A-301.
HOUSE_TOKEN = r"[A-Za-z]?-?\d{1,5}(?:[-/][A-Za-z0-9]{1,6}){0,5}[A-Za-z]?"
HOUSE_MARKER = (
    r"(?i)(?:^|[\s,;#])(?:h\s*\.?\s*no|house\s*no|door\s*no|"
    r"plot\s*no|property\s*no|shop\s*no|flat\s*no|unit\s*no|"
    r"no\.?|n[°º])\s*[:#.-]?\s*(" + HOUSE_TOKEN + r")"
)
HOUSE_SEGMENT = r"(?i)(?:^|[,;]\s*)#?\s*(" + HOUSE_TOKEN + r")(?:\s|[,;]|$)"

INDIA_PIN_LABEL = r"(?i)\bpin(?:\s*code)?\b\s*[:#-]?\s*(\d{6})\b"
INDIA_PIN_SEGMENT = r"(?:^|[,;])\s*(\d{6})\s*(?:[,;]|$)"
INDIA_PIN_CITY_END = r"(?i)[,;]\s*[\p{L}][\p{L}\s-]{1,50}\s+(\d{6})\s*$"
US_ZIP_LABEL = r"(?i)\b(?:zip(?:code)?|postal\s*code)\s*[:#-]?\s*(\d{5}(?:-\d{4})?)\b"
US_ZIP_END = r"(?i)(?:^|[,;])\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\s*$"
FR_STANDALONE = r"(?:^|[,;])\s*(\d{5})\s*(?:[,;]|$)"
FR_CODE_CITY = r"(?i)(?:^|[,;])\s*(\d{5})\s+([\p{L}-]+)"
FR_COMPACT_CEDEX = r"(?i)(?:^|[,;])\s*(\d{5})[\p{L}][\p{L}\s-]{1,50}\s+cedex\b"
FR_CEDEX_CITY = r"(?i)(?:^|[,;])\s*\d{5}\s*([\p{L}][\p{L}\s-]{1,50}?)\s+cedex(?:\s+\d+)?(?:[,;]|$)"
FR_POSTAL_LABEL = r"(?i)\b(?:code\s*postal|postal\s*code|postcode)\b\s*[:#-]?\s*(\d{5})\b"
GENERIC_POSTAL_LABEL = r"(?i)\b(?:postal\s*code|postcode|zip|pin)\b\s*[:#-]?\s*([A-Z0-9-]{4,10})\b"
GENERIC_UK_POSTAL_LABEL = r"(?i)\b(?:postal\s*code|postcode)\b\s*[:#-]?\s*([A-Z]{1,2}\d[A-Z\d]?\s+\d[A-Z]{2})\b"
GENERIC_SIX_DIGIT = r"(?:^|[,;\s])(\d{6})(?:$|[,;\s])"


def postcode_expr(address: pl.Expr, country: pl.Expr) -> pl.Expr:
    """Return a postal code only where the source text supports one."""
    raw = address.cast(pl.String).fill_null("")
    label = raw.str.extract(GENERIC_POSTAL_LABEL, 1)
    india = pl.coalesce(
        raw.str.extract(INDIA_PIN_LABEL, 1),
        raw.str.extract(INDIA_PIN_SEGMENT, 1),
        raw.str.extract(INDIA_PIN_CITY_END, 1),
    )
    us_state = raw.str.extract(US_ZIP_END, 1).str.to_lowercase()
    us_at_end = pl.when(us_state != "fl").then(raw.str.extract(US_ZIP_END, 2)).otherwise(None)
    us = pl.coalesce(raw.str.extract(US_ZIP_LABEL, 1), us_at_end)
    fr_standalone = raw.str.extract(FR_STANDALONE, 1)
    fr_standalone = pl.when(fr_standalone.str.slice(0, 2) != "00").then(fr_standalone).otherwise(None)
    fr_city = raw.str.extract(FR_CODE_CITY, 1)
    fr_next_word = raw.str.extract(FR_CODE_CITY, 2).str.to_lowercase()
    fr_city = (
        pl.when(
            (fr_city.str.slice(0, 2) != "00")
            & ~fr_next_word.is_in(
                ["rue", "r", "av", "ave", "avenue", "bd", "boulevard", "blvd",
                 "route", "rte", "chemin", "ch", "impasse", "imp", "place",
                 "pl", "quai", "q"]
            )
        )
        .then(fr_city)
        .otherwise(None)
    )
    fr_compact = raw.str.extract(FR_COMPACT_CEDEX, 1)
    fr_compact = pl.when(fr_compact.str.slice(0, 2) != "00").then(fr_compact).otherwise(None)
    fr = pl.coalesce(fr_standalone, fr_city, fr_compact)
    key = country.cast(pl.String).fill_null("").str.to_lowercase()
    return (
        pl.when(key == "india").then(india)
        .when(key == "us").then(us)
        .when(key == "france").then(pl.coalesce(fr, raw.str.extract(FR_POSTAL_LABEL, 1)))
        .otherwise(pl.coalesce(raw.str.extract(GENERIC_UK_POSTAL_LABEL, 1), label, raw.str.extract(GENERIC_SIX_DIGIT, 1)))
        .cast(pl.String)
    )


def house_number_expr(address: pl.Expr, postcode: pl.Expr) -> pl.Expr:
    raw = address.cast(pl.String).fill_null("")
    house = pl.coalesce(
        raw.str.extract(HOUSE_MARKER, 1), raw.str.extract(HOUSE_SEGMENT, 1)
    ).str.to_lowercase()
    return (
        pl.when((house == postcode) | (house == ""))
        .then(None)
        .otherwise(house)
        .cast(pl.String)
    )


def city_expr(address: pl.Expr, country: pl.Expr) -> pl.Expr:
    """Return a best-guess city from comma-delimited address components."""
    raw = address.cast(pl.String).fill_null("")
    segments = (
        raw.str.split(",")
        .list.eval(pl.element().str.strip_chars().str.to_lowercase())
        .list.eval(pl.element().filter(~pl.element().is_in(PLACEHOLDERS)))
    )
    last = segments.list.last()
    previous = segments.list.get(-2, null_on_oob=True)
    before_previous = segments.list.get(-3, null_on_oob=True)
    key = country.cast(pl.String).fill_null("").str.to_lowercase()
    is_region = last.is_in(REGION_LABELS) | (
        (key == "us") & last.str.contains(r"(?i)^[a-z]{2}(?:\s+\d{5}(?:-\d{4})?)?$")
    )
    candidate = pl.when(is_region & previous.is_not_null()).then(previous).otherwise(last)

    def plausible(text: pl.Expr) -> pl.Expr:
        stripped = text.str.replace(r"^\d{5,6}\s+", "").str.strip_chars()
        is_street = stripped.str.contains(STREET_WORDS) | (
            (key == "france") & stripped.str.contains(FRANCE_EXTRA_STREET_WORDS)
        )
        return (
            stripped.str.contains(r"\p{L}")
            & ~stripped.str.contains(r"\d")
            & ~is_street
            & (stripped.str.len_chars() <= 60)
            & ~stripped.is_in(REGION_LABELS)
            & ~stripped.is_in(PLACEHOLDERS)
        )

    fallback = pl.when(is_region).then(before_previous).otherwise(previous)
    chosen = (
        pl.when(plausible(candidate)).then(candidate)
        .when(plausible(fallback)).then(fallback)
        .otherwise(None)
    )
    generic_city = clean_text(chosen.str.replace(r"^\d{5,6}\s+", "")).replace("", None)
    cedex_city = clean_text(raw.str.extract(FR_CEDEX_CITY, 1)).replace("", None)
    french_city = (
        strip_accents(pl.coalesce(generic_city, cedex_city))
        .str.replace(r"^st\s+", "saint ")
        .str.replace(r"^ste\s+", "sainte ")
    )
    return (
        pl.when(key == "france").then(french_city)
        .otherwise(generic_city)
        .cast(pl.String)
    )


def address_fields(address: pl.Expr, country: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    postcode = postcode_expr(address, country)
    return postcode, city_expr(address, country), house_number_expr(address, postcode)
