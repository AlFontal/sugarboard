import hashlib

import pytest
from requests.adapters import HTTPAdapter

from src.nightscout_client import NightscoutClient
from src.ui import components


def test_api_secret_hash_uses_nightscout_sha1_protocol():
    client = NightscoutClient("https://nightscout.example", api_secret="secret")

    assert client.api_secret_hash == hashlib.sha1(b"secret").hexdigest()


def test_client_configures_retrying_http_adapters():
    client = NightscoutClient("https://nightscout.example")
    adapter = client.session.get_adapter("https://nightscout.example")

    assert isinstance(adapter, HTTPAdapter)
    assert adapter.max_retries.total == 3
    assert 503 in adapter.max_retries.status_forcelist


def test_sanitize_base_url_defaults_to_https():
    assert components._sanitize_base_url("nightscout.example/") == "https://nightscout.example"


def test_sanitize_base_url_rejects_http_without_opt_in(monkeypatch):
    monkeypatch.setattr(components, "ALLOW_HTTP", False)

    with pytest.raises(ValueError, match="https"):
        components._sanitize_base_url("http://nightscout.example")


def test_sanitize_base_url_rejects_private_addresses(monkeypatch):
    monkeypatch.setattr(components, "ALLOW_INSECURE_NS_URLS", False)

    with pytest.raises(ValueError, match="private/local"):
        components._sanitize_base_url("https://127.0.0.1")
