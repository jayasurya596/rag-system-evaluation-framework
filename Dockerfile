FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies (git, build tools if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data_pipeline/ ./data_pipeline/
COPY evaluation/ ./evaluation/
COPY notebooks/ ./notebooks/
COPY tests/ ./tests/
COPY entrypoint.py .
COPY pyproject.toml .

# Expose Streamlit port
EXPOSE 8501
# Expose FastAPI port
EXPOSE 8000

# Set entrypoint command to run our custom python process supervisor
CMD ["python", "entrypoint.py"]
