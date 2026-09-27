from src.blocking.tfidf_v0 import peak_ram_gb


def test_peak_ram_gb_is_available_on_this_platform():
    assert peak_ram_gb() > 0
