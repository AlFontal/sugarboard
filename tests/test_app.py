import os

import pytest
from seleniumbase import BaseCase


BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8080")


class NiceGUIVisibilityTests(BaseCase):
    """SeleniumBase smoke tests to ensure critical UI blocks render."""

    def open_dashboard(self) -> None:
        self.open(BASE_URL)
        self.wait_for_element("body")

    @pytest.mark.e2e
    def test_core_sections_are_visible(self) -> None:
        self.open_dashboard()
        selectors = [
            ".status-card",
            ".ns-settings-card",
            ".pattern-card",
            ".tir-select",
            ".night-input",
            ".ns-primary-button",
        ]
        for selector in selectors:
            self.wait_for_element(selector)
            self.assert_element_visible(selector)

    @pytest.mark.e2e
    def test_theme_toggle_switch_is_interactive(self) -> None:
        self.open_dashboard()
        toggle_selector = ".theme-toggle-simple .q-toggle__inner, .theme-toggle-simple .q-switch__inner"
        self.wait_for_element(toggle_selector)
        self.assert_element_visible(toggle_selector)
        self.click(toggle_selector)
        self.wait_for_element("body.light-theme")
