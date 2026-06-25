#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Load configurations
PROJECT_ID=$1
REGION=${2:-us-central1}

if [ -z "$PROJECT_ID" ]; then
    echo "Usage: ./deploy.sh <GCP_PROJECT_ID> [GCP_REGION]"
    exit 1
fi

echo "=== STARTING CLOUD RUN DEPLOYMENT ==="
echo "Project ID: $PROJECT_ID"
echo "Region:     $REGION"

# Check for gcloud
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed. Please install it to deploy."
    exit 1
fi

# Check for docker
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed. Please install it to build container images."
    exit 1
fi

# Check for terraform
if ! command -v terraform &> /dev/null; then
    echo "Error: terraform is not installed. Please install it to provision infrastructure."
    exit 1
fi

# Initialize terraform
cd deploy
terraform init

# 1. Target and provision the Artifact Registry repository first
echo "--> Provisioning Artifact Registry repository..."
terraform apply \
  -target=google_artifact_registry_repository.repo \
  -var="project_id=$PROJECT_ID" \
  -var="region=$REGION" \
  -auto-approve

# Get registry repository URL
REPO_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/rag-eval-repo"

# 2. Export local requirements.txt and Build Docker image
echo "--> Building Docker container image locally..."
cd ..
uv pip freeze > requirements.txt
docker build -t "${REPO_URL}/backend:latest" .

# 3. Configure Docker credential helper
echo "--> Configuring Docker authorization for GCP..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# 4. Push image to registry
echo "--> Pushing Docker image to Google Artifact Registry..."
docker push "${REPO_URL}/backend:latest"

# 5. Provision Secret Manager and Cloud Run service
echo "--> Provisioning Secret Manager and Cloud Run Service..."
cd deploy
terraform apply \
  -var="project_id=$PROJECT_ID" \
  -var="region=$REGION" \
  -auto-approve

BACKEND_URL=$(terraform output -raw backend_url)

echo "=================================================="
echo "          DEPLOYMENT COMPLETED SUCCESSFULLY"
echo "=================================================="
echo "Backend service is live at: $BACKEND_URL"
echo ""
echo "Note: You must set your GEMINI_API_KEY secret in the Google Cloud Console Secret Manager"
echo "secret named 'gemini-api-key' for the backend service to access the LLM."
echo "=================================================="
