"""CPU regression for the actual checkpoint migration wrapper."""
import ast
import unittest
from pathlib import Path
import elements

class MigrationContract(unittest.TestCase):
    def test_checkpoint_registration_and_delegation(self):
        source = Path(__file__).resolve().parents[1] / "scripts/continuation_train.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.ClassDef) and n.name == "InitialReplayMigration")
        class Replay:
            def save(self):
                return {"saved": True}
            def migrate_uniform(self, data):
                return {"loaded": data}
        report = {}
        scope = {"replay": Replay(), "migration_report": report}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        wrapper = scope["InitialReplayMigration"]()
        checkpoint = elements.Checkpoint()
        checkpoint.replay = wrapper
        self.assertEqual(wrapper.save(), {"saved": True})
        wrapper.load(123)
        self.assertEqual(report, {"loaded": 123})

if __name__ == "__main__":
    unittest.main()
