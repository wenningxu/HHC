"""Run the migrated cloud runtime on Edmond without root or host path changes."""
from pathlib import Path
import os
import sys

BASE = Path('/mnt/8T-HDD/wenning/hhc_migration_20261020')

def command(arguments):
    return [
        'bwrap', '--tmpfs', '/', '--ro-bind', '/usr', '/usr',
        '--ro-bind', '/etc', '/etc', '--symlink', 'usr/bin', '/bin',
        '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64',
        '--proc', '/proc', '--dev-bind', '/dev', '/dev',
        '--ro-bind', '/sys', '/sys', '--bind', '/home', '/home',
        '--bind', '/mnt', '/mnt', '--bind', '/run', '/run',
        '--dir', '/tmp', '--bind', str(BASE/'workspace'), '/workspace',
        '--dir', '/root', '--setenv', 'HOME', '/root',
        '--setenv', 'MPLBACKEND', 'Agg',
        '--setenv', 'PATH', '/workspace/runtime-bin:/opt/hhc-env/bin:/usr/local/bin:/usr/bin:/bin',
        '--bind', str(BASE/'python38'),
        '/root/.local/share/uv/python/cpython-3.8.20-linux-x86_64-gnu',
        '--bind', str(BASE/'cache'), '/root/.cache',
        '--bind', str(BASE/'hhc-env'), '/opt/hhc-env',
        '--bind', str(BASE/'hhc-vq-deps'), '/opt/hhc-vq-deps',
        '--bind', str(BASE/'hhc-baseline-deps'), '/opt/hhc-baseline-deps',
        '--tmpfs', '/usr/local',
        '--bind', str(BASE/'python312-local'), '/usr/local/lib/python3.12',
        '--bind', str(BASE/'hhc_inputs'), '/dev/shm/hhc_inputs',
        '--chdir', '/workspace/HHC', '--', *arguments,
    ]

if __name__ == '__main__':
    os.execvp('bwrap', command(sys.argv[1:]))
