Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Khởi động Hệ thống Weather MLOps (Local)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Thiết lập PYTHONPATH và đường dẫn tuyệt đối
$base_dir = (Get-Location).Path
$env:PYTHONPATH = $base_dir

# Đường dẫn tới python tuyệt đối (nếu có venv)
$python_cmd = "python"
if (Test-Path "$base_dir\.venv\Scripts\python.exe") {
    $python_cmd = "$base_dir\.venv\Scripts\python.exe"
} elseif (Test-Path "$base_dir\venv\Scripts\python.exe") {
    $python_cmd = "$base_dir\venv\Scripts\python.exe"
} elseif (Test-Path "$base_dir\venv_new\Scripts\python.exe") {
    $python_cmd = "$base_dir\venv_new\Scripts\python.exe"
}

Write-Host "Sử dụng Python: $python_cmd" -ForegroundColor Yellow

# Khởi chạy Data API (Port 8001)
Write-Host "Starting Data API (Port 8001)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.ui.RawUI.WindowTitle='Data API (8001)'; `$env:PYTHONPATH='$base_dir'; cd '$base_dir'; & '$python_cmd' -m uvicorn services.data_api.main:app --port 8001 --reload --env-file .env"

# Khởi chạy Prophet API (Port 8003)
Write-Host "Starting Prophet API (Port 8003)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.ui.RawUI.WindowTitle='Prophet API (8003)'; `$env:PYTHONPATH='$base_dir'; cd '$base_dir'; & '$python_cmd' -m uvicorn services.prophet_api.main:app --port 8003 --reload --env-file .env"

# Khởi chạy LSTM API (Port 8004)
Write-Host "Starting LSTM API (Port 8004)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.ui.RawUI.WindowTitle='LSTM API (8004)'; `$env:PYTHONPATH='$base_dir'; cd '$base_dir'; & '$python_cmd' -m uvicorn services.lstm_api.main:app --port 8004 --reload --env-file .env"

# Khởi chạy Forecast API - Orchestrator (Port 8000)
Write-Host "Starting Forecast API (Port 8000)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.ui.RawUI.WindowTitle='Forecast API (8000)'; `$env:PYTHONPATH='$base_dir'; cd '$base_dir'; & '$python_cmd' -m uvicorn services.forecast_api.main:app --port 8000 --reload --env-file .env"

# Khởi chạy Dashboard (Port 8080)
Write-Host "Starting Dashboard (Port 8080)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.ui.RawUI.WindowTitle='Dashboard (8080)'; cd '$base_dir\services\dashboard_ui'; & '$python_cmd' -m http.server 8080"

Write-Host "`nHoàn tất! Hệ thống đang khởi động trong 5 cửa sổ mới." -ForegroundColor Cyan
Write-Host "Truy cập Dashboard tại: http://localhost:8080" -ForegroundColor Cyan
