import pytest

from src.config import resolve_city_config


@pytest.mark.parametrize("city", ["karachi", " KARACHI ", "Karachi"])
def test_resolve_city_config_normalizes_preset_names(monkeypatch, city):
    monkeypatch.delenv("CITY_LATITUDE", raising=False)
    monkeypatch.delenv("CITY_LONGITUDE", raising=False)
    monkeypatch.delenv("CITY_TIMEZONE", raising=False)

    config = resolve_city_config(city=city)

    assert config.city == "Karachi"
    assert config.country == "Pakistan"
    assert config.latitude == pytest.approx(24.8607)
    assert config.longitude == pytest.approx(67.0011)
    assert config.timezone == "Asia/Karachi"
