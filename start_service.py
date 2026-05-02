"""
Universal service starter script for Cloud Run deployments.
Allows selecting which service to run: dashboard-ui, data-api, forecast-api, prophet-api, lstm-api
"""
import sys
import os
import subprocess

SERVICE = os.environ.get("SERVICE_NAME", "dashboard-ui")

# Service configurations
SERVICES = {
    "dashboard-ui": {
        "module": "services.dashboard_ui.serve",
        "port": 8080,
        "description": "Dashboard HTTP server"
    },
    "data-api": {
        "module": "services.data_api.main:app",
        "port": 8001,
        "cmd": "uvicorn",
        "description": "Data API service"
    },
    "forecast-api": {
        "module": "services.forecast_api.main:app",
        "port": 8000,
        "cmd": "uvicorn",
        "description": "Forecast API service"
    },
    "prophet-api": {
        "module": "services.prophet_api.main:app",
        "port": 8003,
        "cmd": "uvicorn",
        "description": "Prophet API service"
    },
    "lstm-api": {
        "module": "services.lstm_api.main:app",
        "port": 8004,
        "cmd": "uvicorn",
        "description": "LSTM API service"
    },
}

if SERVICE not in SERVICES:
    print(f"ERROR: Unknown service '{SERVICE}'")
    print(f"Available services: {', '.join(SERVICES.keys())}")
    sys.exit(1)

config = SERVICES[SERVICE]
port = os.environ.get("PORT", config["port"])

print(f"[OK] Starting {config['description']}...")

if SERVICE == "dashboard-ui":
    # Run Python script directly
    subprocess.run([sys.executable, "-m", config["module"]], check=False)
else:
    # Run uvicorn for FastAPI services
    cmd = [
        sys.executable, "-m", "uvicorn",
        config["module"],
        "--host", "0.0.0.0",
        "--port", str(port)
    ]
    subprocess.run(cmd, check=False)
