"""
Launcher script that starts the local Guitar AI Studio server and opens the browser.
"""

import sys
import webbrowser
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from server import run_server

PORT = 8080

def on_server_ready(actual_port):
    url = f"http://127.0.0.1:{actual_port}"
    print(f"Opening browser at {url}...")
    webbrowser.open(url)

if __name__ == "__main__":
    run_server(PORT, on_ready=on_server_ready)
