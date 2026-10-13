import unittest
from pathlib import Path

from tools.materialize_scale_v3_protocol import expand


class ScaleV3ProtocolTemplateTest(unittest.TestCase):
    def test_expands_tokens_in_values_and_mapping_keys(self):
        token = chr(36) + "{CRAFT_DATA_ROOT}"
        repo_token = chr(36) + "{REPO_ROOT}"
        source = {token + "/models/start": {
            "checkpoint": token + "/models/start",
            "audit": repo_token + "/docs/audit.json",
        }}
        self.assertEqual(expand(source, Path("/tmp/data"), Path("/tmp/repo")), {
            "/tmp/data/models/start": {
                "checkpoint": "/tmp/data/models/start",
                "audit": "/tmp/repo/docs/audit.json",
            }
        })


if __name__ == "__main__":
    unittest.main()
