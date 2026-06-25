output "repository_url" {
  value       = google_artifact_registry_repository.repo.repository_url
  description = "The Artifact Registry Docker Repository URL"
}

output "backend_url" {
  value       = google_cloud_run_service.backend.status[0].url
  description = "The public URL of the deployed FastAPI RAG backend."
}
