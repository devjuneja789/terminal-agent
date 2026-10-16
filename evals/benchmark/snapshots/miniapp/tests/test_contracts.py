"""Executable acceptance checks referenced by individual benchmark tasks."""

import os
import unittest
from unittest.mock import patch

from shop import api, config, validators


class Contracts(unittest.TestCase):
    def test_timeout_from_environment(self):
        with patch.dict(os.environ, {"REQUEST_TIMEOUT_SECONDS": "25"}):
            self.assertEqual(config.request_timeout(), 25)

    def test_timeout_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.request_timeout(), 15)

    def test_page_size_lower_bound(self):
        self.assertEqual(config.page_size(0), 1)

    def test_page_size_upper_bound(self):
        self.assertEqual(config.page_size(101), 100)

    def test_boolean_is_not_an_id(self):
        self.assertFalse(validators.is_positive_id(True))

    def test_reserved_slug_is_rejected(self):
        self.assertFalse(validators.is_slug("admin"))

    def test_ready_route(self):
        self.assertEqual(api.resolve_route("GET", "/ready"), "readiness")

    def test_item_route_rejects_zero(self):
        self.assertIsNone(api.resolve_route("GET", "/items/0"))


if __name__ == "__main__":
    unittest.main()
