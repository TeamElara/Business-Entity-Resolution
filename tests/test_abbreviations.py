import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.normalize import normalize_df


@pytest.mark.parametrize('country', ['US', 'India', 'France', 'Unseen', None])
def test_fallback_and_field_separation(country):
    df = pl.DataFrame({'business_name': ['ACME Pvt. Ltd. Corp Inc St'],
                       'business_address': ['12 RD., ST AVE Blvd Ln Hwy Bldg Flr Apt Pvt'],
                       'country': [country]})
    result = normalize_df(df)
    assert result['name_norm'][0] == 'acme private limited corporation incorporated st'
    expected_st = 'saint' if country == 'France' else 'street'
    assert result['addr_norm'][0] == f'12 road {expected_st} avenue boulevard lane highway building floor apartment pvt'
    assert_frame_equal(result.select(df.columns), df)
    assert_frame_equal(normalize_df(result), result)


def test_no_substring_replacements():
    text = 'ltdfoo acmecorp incorporated pvt2 streetford st2 broadway avenue'
    result = normalize_df(pl.DataFrame({'business_name': [text], 'business_address': [text]}))
    assert result['name_norm'][0] == text
    assert result['addr_norm'][0] == text


def test_repeated_tokens_and_unicode_boundaries():
    result = normalize_df(pl.DataFrame({'business_name': ['PVT/Pvt LTD-Ltd'],
                                       'business_address': ['Rd Rd सड़कrd rdसड़क']}))
    assert result['name_norm'][0] == 'private private limited limited'
    assert result['addr_norm'][0] == 'road road सड़कrd rdसड़क'
