"""Check dependencies without allocating GPU unless explicitly requested."""
import sys,json,argparse,importlib.metadata,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import project_runtime as rt
p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');a=p.parse_args()
assert sys.version_info[:2]==(3,11),'Python 3.11 required'
for name,version in [('jax','0.4.38'),('jaxlib','0.4.38'),('crafter','1.8.3'),('numpy','1.26.4')]:
    actual=importlib.metadata.version(name)
    assert actual==version,(name,actual,version)
subprocess.run([sys.executable,'-m','pip','check'],check=True)
if a.gpu:
    import jax
    assert any(d.platform=='gpu' for d in jax.devices()),'No usable CUDA JAX device'
print(json.dumps(dict(passed=True,gpu_checked=a.gpu,run=rt.RUN,data_root=str(rt.DATA),python=sys.executable),indent=2))
