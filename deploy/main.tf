terraform {
  required_version = ">= 1.0.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Artifact Registry Repository
resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "rag-eval-repo"
  description   = "Docker repository for RAG API and Dashboard"
  format        = "DOCKER"
}

# 2. Secret Manager for Gemini API Key
resource "google_secret_manager_secret" "api_key" {
  secret_id = "gemini-api-key"
  replication {
    automatic = true
  }
}

# 3. Cloud Run Service for FastAPI Backend
resource "google_cloud_run_service" "backend" {
  name     = "rag-backend"
  location = var.region

  template {
    spec {
      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.repo.repository_id}/backend:latest"
        
        ports {
          container_port = 8080
        }

        # Environment variable reading from Secret Manager
        env {
          name = "GEMINI_API_KEY"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.api_key.secret_id
              key  = "latest"
            }
          }
        }

        # Setup standard CPU/Memory configurations
        resources {
          limits = {
            cpu    = "1000m"
            memory = "2Gi"
          }
        }
      }
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = "10"
        "autoscaling.knative.dev/minScale" = "0"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# 4. Make Backend Publicly Accessible
resource "google_cloud_run_service_iam_member" "backend_public" {
  service  = google_cloud_run_service.backend.name
  location = google_cloud_run_service.backend.location
  role     = "roles/run.viewer"
  member   = "allUsers"
}

# 5. Access Permission for Cloud Run to read the Secret
resource "google_secret_manager_secret_iam_member" "secret_accessor" {
  secret_id = google_secret_manager_secret.api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_cloud_run_service.backend.template[0].spec[0].service_account_name == "" ? "service-${data.google_project.project.number}@serverless-robot-prod.iam.gserviceaccount.com" : google_cloud_run_service.backend.template[0].spec[0].service_account_name}"
}

data "google_project" "project" {}
