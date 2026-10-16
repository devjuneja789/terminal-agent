"""Sanity checks proving the packaged repository snapshot can be imported."""

import unittest


class SnapshotTests(unittest.TestCase):
    def test_package_imports(self):
        import shop.api
        import shop.config
        import shop.users
        import shop.validators

        self.assertTrue(shop.api.ROUTES)
        self.assertGreater(shop.config.MAX_PAGE_SIZE, 0)
        self.assertTrue(shop.validators.is_slug("sample-item"))
        self.assertTrue(shop.users.is_active({"active": True}))
