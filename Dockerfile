FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install the package using pyproject.toml
COPY pyproject.toml poetry.lock* ./
RUN pip install --upgrade pip setuptools wheel && pip install --no-cache-dir .

# Copy application code
COPY . .

# Expose default port (adjust if your app uses a different one)
EXPOSE 8000

# Default command — run the app with Python. Override in docker run if needed.
CMD ["python", "-u", "main.py"]
