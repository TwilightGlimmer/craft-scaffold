"""Read-only scan of staged Git blobs. Run through python.sh before publishing."""
import hashlib,pathlib,re,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
# Only the three manifest-verified teaching episodes are allowed as binary media.
APPROVED_MEDIA={
 'docs/course-report/project-materials/videos/baseline-full-episode.mp4':('bb5968803ad8e433d070ba70ec6dec15db75715536e612cbe80ff394726aa8d4',10*1024*1024),
 'docs/course-report/project-materials/videos/branch-full-episode.mp4':('492b2326f263c97a416c173c0aaf488ed4c0644049bc870c549349c5cff75ccd',10*1024*1024),
 'docs/course-report/project-materials/videos/source-full-episode.mp4':('feececa4b6f2eadc89b7b5a2810eabe940fb384609755e74303cc22fdfa6e0dd',10*1024*1024),
}
def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])
entries=git('ls-files','--stage','-z').decode().split('\0')
issues=[]; count=0; total=0
patterns=[
 ('personal server path',r'/root/gpufree-data/[A-Za-z0-9_-]+'),
 ('fixed GPU UUID',r'GPU-[0-9a-f]{8}-[0-9a-f-]{27,}'),
 ('Windows user path',r'C:[/\\]Users[/\\][^/\\\s]+'),
 ('private key',r'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----'),
 ('GitHub token',r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
 ('AWS access key',r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
 ('credential URL',r'https?://[^/\s:@]+:[^/\s@]+@'),
 ('assigned secret',r'''(?im)^\s*["']?(?:api[_-]?key|access[_-]?token|password|secret[_-]?key)["']?\s*[:=]\s*["'][^"'\s]{8,}["']'''),
]
for entry in filter(None,entries):
    meta,name=entry.split('\t',1); mode,oid,stage=meta.split()
    count+=1
    if mode not in ('100644','100755'):issues.append((name,'symlink or gitlink'));continue
    data=git('cat-file','blob',oid);total+=len(data)
    approved_media=APPROVED_MEDIA.get(name)
    if len(data)>2*1024*1024 and not approved_media:issues.append((name,'larger than 2 MiB'))
    if approved_media:
        expected_hash,max_bytes=approved_media
        if len(data)>max_bytes:issues.append((name,'approved media exceeds 10 MiB'))
        if data[4:8]!=b'ftyp':issues.append((name,'invalid MP4 container'))
        if hashlib.sha256(data).hexdigest()!=expected_hash:issues.append((name,'approved media SHA256 mismatch'))
    try:text=data.decode('utf-8')
    except UnicodeDecodeError:
        if name.startswith('docs/assets/') and name.endswith('.png') and data.startswith(b'\x89PNG\r\n\x1a\n'):
            from PIL import Image
            import io
            try:
                im=Image.open(io.BytesIO(data)); im.verify()
            except Exception:
                issues.append((name,'invalid PNG illustration'))
        elif approved_media and data[4:8]==b'ftyp':
            pass
        else:
            issues.append((name,'unapproved binary file'))
        continue
    if name!= 'tools/check_publication.py':
        for label,pattern in patterns:
            if re.search(pattern,text):issues.append((name,label))
    parts=pathlib.PurePosixPath(name).parts
    if any(x in {'cache','logs','tmp','.deps','.bootstrap','.versions','.diagnostics','.ssh','models','checkpoints','__pycache__'} for x in parts):
        issues.append((name,'runtime/private directory'))
    if pathlib.PurePosixPath(name).suffix.lower() in {'.pkl','.pt','.pth','.pem','.key','.mp4','.gif','.npz'} and not approved_media:
        issues.append((name,'artifact or credential extension'))
if not count:issues.append(('<index>','no staged files'))
print('Scanned %d indexed files; %d bytes; %d findings.'%(count,total,len(issues)))
for name,reason in issues:print(name+': '+reason)
sys.exit(bool(issues))
