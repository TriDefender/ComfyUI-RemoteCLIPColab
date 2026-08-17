"""Tunnel management for the Remote CLIP Colab runtime.

Providers:
  cloudflare - quick tunnel via the cloudflared binary (no account needed)
  ngrok      - ngrok binary with a user supplied authtoken
  direct     - no tunnel; the server binds a public interface itself
"""
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

CLOUDFLARED_URL = {
    ("Linux", "x86_64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    ("Linux", "aarch64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
    ("Windows", "AMD64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
    ("Darwin", "x86_64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz",
    ("Darwin", "arm64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64.tgz",
}
URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
NGROK_RE = re.compile(r"url=(https://[a-zA-Z0-9.\-]+\.ngrok(?:-free)?\.app)")


def log(msg):
    print(f"[tunnel {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _download(url, dest):
    log(f"Downloading {url}")
    tmp = dest + ".tmp"
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    os.replace(tmp, dest)
    os.chmod(dest, 0o755)


def _ensure_cloudflared():
    found = shutil.which("cloudflared")
    if found:
        return found
    key = (platform.system(), platform.machine())
    if key not in CLOUDFLARED_URL:
        raise RuntimeError(f"No cloudflared build for {key}; use --tunnel direct or ngrok")
    bin_dir = os.path.join(os.path.expanduser("~"), ".rcp", "bin")
    os.makedirs(bin_dir, exist_ok=True)
    dest = os.path.join(bin_dir, "cloudflared")
    if platform.system() == "Windows":
        dest += ".exe"
    if not os.path.isfile(dest):
        _download(CLOUDFLARED_URL[key], dest)
    return dest


def _ensure_ngrok():
    found = shutil.which("ngrok")
    if found:
        return found
    machine = platform.machine()
    arch = "arm64" if machine == "aarch64" else "amd64"
    url = f"https://bin.equinox.io/c/bNyf1FQFkfk/ngrok-v3-stable-{platform.system().lower()}-{arch}"
    bin_dir = os.path.join(os.path.expanduser("~"), ".rcp", "bin")
    os.makedirs(bin_dir, exist_ok=True)
    dest = os.path.join(bin_dir, "ngrok")
    if platform.system() == "Windows":
        dest += ".exe"
    if not os.path.isfile(dest):
        _download(url, dest + ".tmp_download")
        # equinox serves a tgz; unpack if needed
        with open(dest + ".tmp_download", "rb") as f:
            head = f.read(2)
        if head == b"\x1f\x8b":
            subprocess.run([sys.executable, "-c",
                            f"import tarfile;tarfile.open({dest + '.tmp_download'!r},'r:gz').extractall({bin_dir!r})"],
                           check=True)
            os.remove(dest + ".tmp_download")
        else:
            os.replace(dest + ".tmp_download", dest)
            os.chmod(dest, 0o755)
    return dest


class TunnelManager:
    """Runs one tunnel process and keeps it alive; exposes the current public URL."""

    def __init__(self, provider, port, host, ngrok_token=None):
        self.provider = provider
        self.port = port
        self.host = host
        self.ngrok_token = ngrok_token
        self.public_url = None
        self.status = "starting"
        self.last_error = None
        self._proc = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self.provider == "direct":
            self.public_url = f"http://{self.host}:{self.port}"
            self.status = "direct"
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except OSError:
                pass

    def info(self):
        return {
            "provider": self.provider,
            "public_url": self.public_url,
            "status": self.status,
            "last_error": self.last_error,
        }

    def _run(self):
        while not self._stop.is_set():
            try:
                if self.provider == "cloudflare":
                    self._run_cloudflared()
                elif self.provider == "ngrok":
                    self._run_ngrok()
                else:
                    raise RuntimeError(f"unknown tunnel provider: {self.provider}")
            except Exception as e:  # noqa: BLE001 - keep the supervisor alive
                self.last_error = str(e)
                self.status = "error"
                log(f"tunnel error: {e}; restarting in 5s")
                self._stop.wait(5)
            if self._stop.is_set():
                break
            self.status = "restarting"
            self._stop.wait(2)

    def _spawn(self, cmd):
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        url = None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line:
                log(line)
            if url is None:
                m = URL_RE.search(line) or NGROK_RE.search(line)
                if m:
                    url = m.group(0)
                    self.public_url = url
                    self.status = "up"
                    log(f"PUBLIC URL: {url}")
        if url is None:
            self._proc.wait(timeout=10)
            raise RuntimeError("tunnel exited before producing a public URL")
        self._proc.wait()
        raise RuntimeError("tunnel process exited; restarting")

    def _run_cloudflared(self):
        binary = _ensure_cloudflared()
        cmd = [binary, "tunnel", "--url", f"http://{self.host}:{self.port}",
               "--no-autoupdate", "--protocol", "http2"]
        self._spawn(cmd)

    def _run_ngrok(self):
        if not self.ngrok_token:
            raise RuntimeError("ngrok requires --ngrok-token")
        binary = _ensure_ngrok()
        subprocess.run([binary, "config", "add-authtoken", self.ngrok_token],
                       check=False, capture_output=True)
        cmd = [binary, "http", str(self.port), "--log=stdout"]
        self._spawn(cmd)
