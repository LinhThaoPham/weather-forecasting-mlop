param(
    [string]$ProjectId = "weather-forecasting-494811",
    [string]$Region = "asia-southeast1",
    [string]$BucketName = "weather-forecasting-494811-models",
    [string]$ArtifactRepo = "weather-forecasting",
    [string]$ImageName = "weather-forecasting-mlops",
    [string]$Dataset = "weather_mlops",
    [string]$Table = "weather_historical"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$imageUri = "$Region-docker.pkg.dev/$ProjectId/$ArtifactRepo/${ImageName}:latest"

Write-Host "==> Configure gcloud project"
gcloud config set project $ProjectId | Out-Null
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com storage.googleapis.com bigquery.googleapis.com cloudscheduler.googleapis.com aiplatform.googleapis.com

Write-Host "==> Create GCS bucket (if missing)"
if (-not (gcloud storage buckets list --project $ProjectId --format="value(name)" | Select-String -SimpleMatch $BucketName)) {
    gcloud storage buckets create "gs://$BucketName" --project $ProjectId --location $Region --uniform-bucket-level-access
}

Write-Host "==> Upload existing model artifacts"
gcloud storage cp --recursive ".\models\current\*" "gs://$BucketName/models/current/" 2>$null
if (Test-Path ".\models\registry.json") { gcloud storage cp ".\models\registry.json" "gs://$BucketName/models/registry.json" }
if (Test-Path ".\models\training_history.json") { gcloud storage cp ".\models\training_history.json" "gs://$BucketName/models/training_history.json" }

Write-Host "==> Create BigQuery dataset/table"
$datasetExists = bq ls --project_id=$ProjectId --format=prettyjson | Select-String -SimpleMatch "`"datasetId`": `"$Dataset`""
if (-not $datasetExists) {
    bq --location=$Region mk --dataset "${ProjectId}:$Dataset"
}
$tableFqn = "$ProjectId.$Dataset.$Table"
$createTableSql = @'
CREATE TABLE IF NOT EXISTS `%TABLE_FQN%` (
  city_id STRING NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  temperature FLOAT64,
  humidity FLOAT64,
  cloud_cover FLOAT64,
  apparent_temp FLOAT64,
  precipitation FLOAT64,
  rain FLOAT64,
  weather_code INT64,
  pressure FLOAT64,
  wind_speed FLOAT64,
  wind_direction FLOAT64,
  wind_gusts FLOAT64,
  dewpoint FLOAT64,
  fetched_at TIMESTAMP
)
PARTITION BY DATE(timestamp)
CLUSTER BY city_id
'@
$createTableSql = $createTableSql.Replace("%TABLE_FQN%", $tableFqn)
$createTableSql | bq query --use_legacy_sql=false

Write-Host "==> Build + push image to Artifact Registry"
$repoExists = gcloud artifacts repositories describe $ArtifactRepo --location=$Region --format="value(name)" 2>$null
if (-not $repoExists) {
    gcloud artifacts repositories create $ArtifactRepo --repository-format=docker --location=$Region --description="Weather forecasting images"
}
gcloud auth configure-docker "$Region-docker.pkg.dev" --quiet
gcloud builds submit --tag $imageUri .

Write-Host "==> Deploy Cloud Run services"
gcloud run deploy data-api --image $imageUri --region $Region --platform managed --allow-unauthenticated `
  --port 8080 --command uvicorn --args services.data_api.main:app,--host,0.0.0.0,--port,8080 `
  --set-env-vars "USE_BIGQUERY=true,GCP_PROJECT_ID=$ProjectId,GCP_REGION=$Region,BIGQUERY_DATASET=$Dataset,BIGQUERY_HISTORICAL_TABLE=$Table,USE_GCS=true,GCS_BUCKET_NAME=$BucketName"

gcloud run deploy prophet-api --image $imageUri --region $Region --platform managed --allow-unauthenticated `
  --port 8080 --command uvicorn --args services.prophet_api.main:app,--host,0.0.0.0,--port,8080 `
  --set-env-vars "USE_GCS=true,GCP_PROJECT_ID=$ProjectId,GCP_REGION=$Region,GCS_BUCKET_NAME=$BucketName"

gcloud run deploy lstm-api --image $imageUri --region $Region --platform managed --allow-unauthenticated `
  --port 8080 --command uvicorn --args services.lstm_api.main:app,--host,0.0.0.0,--port,8080 `
  --set-env-vars "USE_GCS=true,GCP_PROJECT_ID=$ProjectId,GCP_REGION=$Region,GCS_BUCKET_NAME=$BucketName"

$prophetUrl = gcloud run services describe prophet-api --region $Region --format="value(status.url)"
$lstmUrl = gcloud run services describe lstm-api --region $Region --format="value(status.url)"

gcloud run deploy forecast-api --image $imageUri --region $Region --platform managed --allow-unauthenticated `
  --port 8080 --command uvicorn --args services.forecast_api.main:app,--host,0.0.0.0,--port,8080 `
  --set-env-vars "PROPHET_API_URL=$prophetUrl,LSTM_API_URL=$lstmUrl,USE_GCS=true,GCS_BUCKET_NAME=$BucketName,GCP_PROJECT_ID=$ProjectId,GCP_REGION=$Region"

Write-Host "==> Create Cloud Run retrain job"
gcloud run jobs deploy retrain-job --image $imageUri --region $Region `
  --command python --args daily_pipeline.py `
  --set-env-vars "USE_GCS=true,USE_BIGQUERY=true,ENABLE_VERTEX_METRICS=true,GCP_PROJECT_ID=$ProjectId,GCP_REGION=$Region,GCS_BUCKET_NAME=$BucketName,BIGQUERY_DATASET=$Dataset,BIGQUERY_HISTORICAL_TABLE=$Table"

Write-Host "==> Create Cloud Scheduler job at 2AM VN"
$schedulerSaName = "scheduler-runner"
$schedulerSa = "$schedulerSaName@$ProjectId.iam.gserviceaccount.com"
$schedulerSaExists = gcloud iam service-accounts describe $schedulerSa --project $ProjectId --format="value(email)" 2>$null
if (-not $schedulerSaExists) {
    gcloud iam service-accounts create $schedulerSaName --project $ProjectId --display-name "Cloud Scheduler Run Job Caller"
}
gcloud projects add-iam-policy-binding $ProjectId --member "serviceAccount:$schedulerSa" --role "roles/run.invoker" | Out-Null
$runJobUri = "https://run.googleapis.com/v2/projects/$ProjectId/locations/$Region/jobs/retrain-job:run"
gcloud scheduler jobs delete retrain-2am-vn --location $Region --quiet 2>$null
gcloud scheduler jobs create http retrain-2am-vn `
  --location $Region `
  --schedule "0 2 * * *" `
  --time-zone "Asia/Ho_Chi_Minh" `
  --uri $runJobUri `
  --http-method POST `
  --oauth-service-account-email $schedulerSa

Write-Host "✅ GCP deployment complete."
