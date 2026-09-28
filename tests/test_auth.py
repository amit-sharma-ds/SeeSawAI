import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from cctv_ai import api, auth


class ApiKeyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.key_file = Path(self.tmp.name) / "api_keys.txt"
        self.patches = [
            mock.patch.object(auth, "KEY_FILE", self.key_file),
            mock.patch.dict(os.environ, {"CCTV_API_KEYS": ""}),
        ]
        for p in self.patches:
            p.start()
        auth._file_cache.update(token=None, keys={})
        self.client = TestClient(api.app)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        auth._file_cache.update(token=None, keys={})
        self.tmp.cleanup()

    def test_open_when_no_keys_exist(self):
        self.assertEqual(self.client.get("/health").json()["auth"], "open")
        self.assertEqual(self.client.get("/cameras").status_code, 200)

    def test_env_key_protects_everything_but_health_and_docs(self):
        with mock.patch.dict(os.environ, {"CCTV_API_KEYS": "secret-one, secret-two"}):
            self.assertEqual(self.client.get("/health").json()["auth"], "api_key")
            self.assertEqual(self.client.get("/health").status_code, 200)
            self.assertEqual(self.client.get("/docs").status_code, 200)
            self.assertEqual(self.client.get("/openapi.json").status_code, 200)

            denied = self.client.get("/cameras")
            self.assertEqual(denied.status_code, 401)
            self.assertIn("X-API-Key", denied.json()["detail"])
            self.assertEqual(self.client.get("/cameras", headers={"X-API-Key": "wrong"}).status_code, 401)
            self.assertEqual(self.client.get("/alerts", headers={"X-API-Key": "secret-two"}).status_code, 200)
            self.assertEqual(self.client.get("/cameras", params={"api_key": "secret-one"}).status_code, 200)
            self.assertEqual(self.client.get("/cameras/none/snapshot", params={"api_key": "secret-one"}).status_code, 404)

    def test_file_keys_with_labels_and_hot_reload(self):
        key = auth.add_key("frontend-team")
        self.assertEqual(auth.load_keys(), {key: "frontend-team"})
        self.assertEqual(self.client.get("/cameras").status_code, 401)
        self.assertEqual(self.client.get("/cameras", headers={"X-API-Key": key}).status_code, 200)

        second = auth.add_key()
        self.assertEqual(auth.load_keys()[second], "")
        self.assertEqual(self.client.get("/cameras", headers={"X-API-Key": second}).status_code, 200)

        self.key_file.write_text(f"# revoked the first key\n{second} mobile\n", encoding="utf-8")
        self.assertEqual(auth.load_keys(), {second: "mobile"})
        self.assertEqual(self.client.get("/cameras", headers={"X-API-Key": key}).status_code, 401)
        self.assertEqual(self.client.get("/cameras", headers={"X-API-Key": second}).status_code, 200)

        self.key_file.unlink()
        self.assertEqual(auth.load_keys(), {})
        self.assertEqual(self.client.get("/cameras").status_code, 200)

    def test_generated_keys_are_long_and_unique(self):
        keys = {auth.generate_key() for _ in range(50)}
        self.assertEqual(len(keys), 50)
        self.assertTrue(all(len(k) >= 40 for k in keys))

    def test_docs_advertise_the_security_schemes(self):
        schemes = self.client.get("/openapi.json").json()["components"]["securitySchemes"]
        self.assertEqual(schemes["APIKeyHeader"]["name"], "X-API-Key")
        self.assertEqual(schemes["APIKeyQuery"]["name"], "api_key")

    def test_cors_headers_are_present(self):
        response = self.client.options(
            "/alerts",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "*")


if __name__ == "__main__":
    unittest.main()
