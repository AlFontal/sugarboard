import os

import pytest
from seleniumbase import BaseCase

RUN_E2E = os.environ.get("RUN_E2E", "0").lower() in {"1", "true", "yes"}

pytestmark = pytest.mark.skipif(not RUN_E2E, reason="Set RUN_E2E=1 to enable Selenium tests")


@pytest.fixture(scope="class", autouse=True)
def _provide_server_url(request, nicegui_server):
    request.cls.server_url = nicegui_server


class NiceGUIVisibilityTests(BaseCase):
    """SeleniumBase smoke tests to ensure critical UI blocks render."""

    server_url: str

    @pytest.mark.e2e
    def test_core_sections_are_visible(self):
        self.open(self.server_url)
        self.wait_for_element("body")
        selectors = [
            ".status-card",
            ".ns-settings-card",
            ".pattern-card",
            ".tir-select",
            ".night-input",
        ]
        for selector in selectors:
            self.wait_for_element(selector)
            self.assert_element_visible(selector)

    @pytest.mark.e2e
    def test_theme_toggle_switch_is_interactive(self) -> None:
        self.open(self.server_url)
        toggle_selector = ".theme-toggle-simple .q-toggle__inner, .theme-toggle-simple .q-switch__inner"
        self.wait_for_element(toggle_selector)
        self.assert_element_visible(toggle_selector)
        self.click(toggle_selector)
        self.wait_for_element("body.light-theme")
