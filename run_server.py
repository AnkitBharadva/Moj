"""
24/7 API Server Launcher.

Runs the FastAPI classification service via Uvicorn.
Usage:
    python run_server.py
"""

import sys
import uvicorn

if __name__ == "__main__":
    port = 8000
    host = "127.0.0.1"
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    if len(sys.argv) > 2:
        host = sys.argv[2]

    print(f"Starting Comment Analysis API 24/7 service on http://{host}:{port}")
    uvicorn.run("src.api:app", host=host, port=port, log_level="info", access_log=True)
