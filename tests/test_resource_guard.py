import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import project_runtime as rt
import resource_guard as guard
class GuardTests(unittest.TestCase):
    def test_busy_gpu_never_launches(self):
        with patch.object(guard,'gpu_query',return_value='123'),patch.object(guard.subprocess,'Popen') as launch:
            with self.assertRaises(RuntimeError):guard.run(['ignored'],Path('unused'),Path('unused'))
            launch.assert_not_called()
    def test_resource_limit_stops_process_group(self):
        rt.DATA.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=rt.DATA) as d:
            child=Mock(pid=123,returncode=-15)
            child.poll.return_value=None
            with patch.object(guard,'gpu_query',side_effect=['','21000']),patch.object(guard.subprocess,'Popen',return_value=child),patch.object(guard.psutil,'Process') as process,patch.object(guard.time,'sleep'),patch.object(guard.os,'killpg') as kill:
                process.return_value.create_time.return_value=1
                process.return_value.children.return_value=[]
                process.return_value.memory_info.return_value.rss=1
                result=guard.run(['ignored'],Path(d)/'test.log',Path(d)/'progress')
                self.assertEqual(result['reason'],'resource_limit')
                kill.assert_called_once_with(123,guard.signal.SIGTERM)
                child.wait.assert_called_once()
    def test_callback_failure_also_stops_child(self):
        rt.DATA.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=rt.DATA) as d:
            child=Mock(pid=123);child.poll.return_value=None
            with patch.object(guard,'gpu_query',return_value=''),patch.object(guard.subprocess,'Popen',return_value=child),patch.object(guard.psutil,'Process'),patch.object(guard.os,'killpg') as kill:
                with self.assertRaises(ValueError):
                    guard.run(['ignored'],Path(d)/'test.log',Path(d)/'progress',Mock(side_effect=ValueError('callback')))
                kill.assert_called_once()
