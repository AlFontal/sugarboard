from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests


class NightscoutClient:
    """Lightweight helper for interacting with a Nightscout instance."""

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: int = 10,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token or None
        self.timeout = timeout

        self.api_secret_hash = None
        if not self.token and api_secret:
            self.api_secret_hash = hashlib.sha1(api_secret.encode("utf-8")).hexdigest()

    def _build_url(self, path: str) -> str:
        return urljoin(f"{self.base_url}/", path.lstrip("/"))

    def _auth_params_headers(self) -> tuple[Dict[str, Any], Dict[str, str]]:
        params: Dict[str, Any] = {}
        headers: Dict[str, str] = {}
        if self.token:
            params["token"] = self.token
        elif self.api_secret_hash:
            headers["api-secret"] = self.api_secret_hash
        return params, headers

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        req_params = dict(params or {})
        auth_params, headers = self._auth_params_headers()
        req_params.update(auth_params)
        response = requests.get(
            self._build_url(path),
            params=req_params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_sgv(self, count: int = 288, extra_find: Optional[Dict[str, Any]] = None) -> Any:
        params: Dict[str, Any] = {"count": count}
        if extra_find:
            self._apply_find(params, extra_find)
        return self._get("/api/v1/entries/sgv.json", params=params)

    @staticmethod
    def _apply_find(params: Dict[str, Any], find_filters: Dict[str, Any]) -> None:
        for key, value in find_filters.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    params[f"find[{key}][{sub_key}]"] = sub_value
            else:
                params[f"find[{key}]"] = value


__all__ = ["NightscoutClient"]
