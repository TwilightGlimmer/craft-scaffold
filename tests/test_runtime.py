import unittest,sys,os,tempfile,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import project_runtime as rt
class RuntimeTests(unittest.TestCase):
    def test_output_containment(self):
        self.assertTrue(rt.owned(rt.DATA/'models/example').is_relative_to(rt.DATA))
        for path in [rt.DATA,rt.DATA/'..'/'escape']:
            with self.assertRaises(ValueError):rt.owned(path)
    def test_symlink_escape(self):
        rt.DATA.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=rt.DATA) as d:
            link=Path(d)/'link';link.symlink_to(rt.DATA.parent,target_is_directory=True)
            with self.assertRaises(ValueError):rt.owned(link/'escape')
    def test_freeze_refuses_changes(self):
        rt.DATA.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=rt.DATA) as d:
            with patch.object(rt,'REPORTS',Path(d)),patch.object(rt,'sources',return_value={'source':'one'}),patch.object(rt,'environment',return_value={'python':'test'}):
                rt.freeze();rt.verify()
                with patch.object(rt,'sources',return_value={'source':'two'}):
                    with self.assertRaises(RuntimeError):rt.freeze()
                    with self.assertRaises(RuntimeError):rt.verify()
    def test_source_manifest_relative(self):
        entries=rt.sources()
        self.assertIn('scripts/curriculum_train.py',entries)
        self.assertIn('configs/train.yaml',entries)
        self.assertTrue(all(not Path(k).is_absolute() for k in entries))
if __name__=='__main__':unittest.main()
