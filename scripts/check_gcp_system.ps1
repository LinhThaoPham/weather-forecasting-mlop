param(
    [string]$ProjectId = "weather-forecasting-494811",
    [string]$Region = "asia-southeast1",
    [string]$BucketName = "weather-forecasting-494811-models",
    [string]$Dataset = "weather_mlops",
    [string]$Table = "weather_historical"
)

$ErrorActionPreference = "Continue"

Write-Host "== Cloud Run services =="
gcloud run services list --platform managed --region $Region --format="table(metadata.name,status.url,status.conditions[0].status)"

Write-Host "`n== Health checks =="
$services = @("data-api", "prophet-api", "lstm-api", "forecast-api")
foreach ($svc in $services) {
    $url = gcloud run services describe $svc --region $Region --format="value(status.url)"
    if ($url) {
        try {
            $resp = Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 20
            Write-Host "$svc => $($resp.StatusCode)"
        } catch {
            Write-Host "$svc => ERROR: $($_.Exception.Message)"
        }
    }
}

Write-Host "`n== GCS artifacts =="
gcloud storage ls "gs://$BucketName/models/current/" 2>$null

Write-Host "`n== BigQuery rows =="
$tableFqn = "$ProjectId.$Dataset.$Table"
bq query --nouse_legacy_sql --format=prettyjson "SELECT COUNT(*) AS row_count FROM ``$tableFqn``"

Write-Host "`n== Scheduler jobs =="
gcloud scheduler jobs list --location $Region --format="table(name,schedule,timeZone,state)"

Write-Host "`n== Vertex experiments =="
gcloud beta ai experiments list --region $Region --format="table(name,createTime)"
