"""One child process group, GPU occupancy check and bounded resources."""
import subprocess,time,os,signal,psutil
from pathlib import Path
import project_runtime as rt
def gpu_query(field,apps=False):
    flag='--query-compute-apps=' if apps else '--query-gpu='
    return subprocess.check_output(['nvidia-smi','-i',rt.GPU,flag+field,'--format=csv,noheader,nounits'],text=True).strip()
def run(cmd,log,progress,on_status=lambda **kw:None):
    if gpu_query('pid',True):raise RuntimeError('Selected GPU already has a compute process')
    start=time.time();peakrss=peakgpu=0;reason='exit'
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as out:
        child=subprocess.Popen(cmd,cwd=rt.ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            on_status(child_pid=child.pid,child_created=psutil.Process(child.pid).create_time(),log=str(log))
            while child.poll() is None:
                time.sleep(5)
                if child.poll() is not None:break
                try:
                    q=psutil.Process(child.pid)
                    rss=0
                    for proc in [q]+q.children(recursive=True):
                        try:rss+=proc.memory_info().rss
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:continue
                mem=int(gpu_query('memory.used'))
                peakrss=max(peakrss,rss);peakgpu=max(peakgpu,mem)
                age=time.time()-max(start,progress.stat().st_mtime if progress.exists() else start)
                on_status(heartbeat=time.time(),rss_bytes=rss,gpu_mib=mem,progress_age=age)
                if rss>16*2**30 or mem>20480:reason='resource_limit'
                elif age>360:reason='stalled'
                elif time.time()-start>2400:reason='attempt_timeout'
                if reason!='exit':break
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    return dict(exit_code=child.returncode,reason=reason,elapsed=time.time()-start,
                peak_rss_bytes=peakrss,peak_gpu_mib=peakgpu,log=str(log))
