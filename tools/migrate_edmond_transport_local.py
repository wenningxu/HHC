"""SSH transport for the user-authorized Edmond migration; no local model transfer."""
import subprocess
import sys

PYTHON = sys.executable
EDMOND = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "newstation"]
RUNPOD = ["ssh", "-i", r"C:\Users\111\.ssh\id_ed25519_newstation",
          "-p", "33923", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
          "root@157.157.221.29"]

def remote(host, code, timeout=90):
    return subprocess.run((EDMOND if host == "edmond" else RUNPOD) + ["python3 -"],
                          input=code, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)

if __name__ == "__main__":
    result = remote(sys.argv[1], sys.stdin.read(), timeout=300)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    raise SystemExit(result.returncode)
