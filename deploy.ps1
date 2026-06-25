param (
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-central1"
)

$ErrorActionPreference = "Stop"

Write-Host "=== STARTING CLOUD RUN DEPLOYMENT (POWERSHELL) ===" -ForegroundColor Cyan
Write-Host "Project ID: $ProjectId"
Write-Host "Region:     $Region"

# Check tools
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "Error: gcloud CLI is not installed. Please install it to deploy."
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Error: docker is not installed. Please install it to build container images."
}
if (-not (Get-Command terraform -ErrorAction SilentlyContinue)) {
    Write-Error "Error: terraform is not installed. Please install it to provision infrastructure."
}

# Navigate to deploy directory
Set-Location deploy
terraform init

# 1. Provision Artifact Registry repository first
Write-Host "--> Provisioning Artifact Registry repository..." -ForegroundColor Green
terraform apply `
  -target=google_artifact_registry_repository.repo `
  -var="project_id=$ProjectId" `
  -var="region=$Region" `
  -auto-approve

$RepoUrl = "${Region}-docker.pkg.dev/${ProjectId}/rag-eval-repo"

# 2. Build Docker image
Write-Host "--> Building Docker container image locally..." -ForegroundColor Green
Set-Location ..
uv pip freeze > requirements.txt
docker build -t "${RepoUrl}/backend:latest" .

# 3. Configure Docker credential helper
Write-Host "--> Configuring Docker authorization for GCP..." -ForegroundColor Green
gcloud auth configure-docker "${Region}-docker.pkg.dev" --quiet

# 4. Push image to registry
Write-Host "--> Pushing Docker image to Google Artifact Registry..." -ForegroundColor Green
docker push "${RepoUrl}/backend:latest"

# 5. Provision Secret Manager and Cloud Run service
Write-Host "--> Provisioning Secret Manager and Cloud Run Service..." -ForegroundColor Green
Set-Location deploy
terraform apply `
  -var="project_id=$ProjectId" `
  -var="region=$Region" `
  -auto-approve

$BackendUrl = terraform output -raw backend_url

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "          DEPLOYMENT COMPLETED SUCCESSFULLY" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Backend service is live at: $BackendUrl"
Write-Host ""
Write-Host "Note: You must set your GEMINI_API_KEY secret in the Google Cloud Console Secret Manager"
Write-Host "secret named 'gemini-api-key' for the backend service to access the LLM."
Write-Host "==================================================" -ForegroundColor Cyan
