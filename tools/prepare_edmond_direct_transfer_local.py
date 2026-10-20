"""Prepare a temporary, read-only, source-restricted cloud-to-cloud rsync key."""
import json
from migrate_edmond_transport_local import remote

setup = remote("edmond", """
from pathlib import Path
import subprocess,json
base=Path('/mnt/8T-HDD/wenning/hhc_migration_20261020')
base.mkdir(exist_ok=True)
key=base/'transfer_key'
if not key.exists():
 subprocess.run(['ssh-keygen','-t','ed25519','-N','','-C','hhc-edmond-migration-20261020','-f',str(key)],check=True,stdout=subprocess.DEVNULL)
key.chmod(0o600)
print(json.dumps({'public_key':key.with_suffix('.pub').read_text().strip()}))
""")
assert setup.returncode == 0, setup.stderr
pub = json.loads(setup.stdout)["public_key"]
source = remote("runpod", """
from pathlib import Path
import json
auth=Path('/root/.ssh/authorized_keys')
pub=PUBLIC_KEY
line='restrict,command="/usr/bin/rrsync -ro /" '+pub
assert Path('/usr/bin/rrsync').exists()
old=auth.read_text()
if pub not in old:
 auth.write_text(old.rstrip()+'\\n'+line+'\\n')
 auth.chmod(0o600)
print(json.dumps({'host_key':Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text().strip()}))
""".replace("PUBLIC_KEY", repr(pub)))
assert source.returncode == 0, source.stderr
hostkey = json.loads(source.stdout)["host_key"]
dest = remote("edmond", """
from pathlib import Path
import subprocess,json
base=Path('/mnt/8T-HDD/wenning/hhc_migration_20261020')
(base/'known_hosts').write_text('[157.157.221.29]:33923 '+HOST_KEY+'\\n')
ssh='ssh -i '+str(base/'transfer_key')+' -p 33923 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile='+str(base/'known_hosts')
p=subprocess.run(['rsync','--list-only','-e',ssh,'root@157.157.221.29:/workspace/HHC/AGENTS.md'],capture_output=True,text=True)
print(json.dumps({'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
assert p.returncode==0
""".replace("HOST_KEY", repr(hostkey)))
print(dest.stdout, dest.stderr)
assert dest.returncode == 0
