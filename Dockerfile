# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Prevent Python from writing .pyc files and ensure output is flushed
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a non-root user and switch to it for better security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose the default port FastAPI/uvicorn serves on
EXPOSE 8000

# Start the FastAPI app using uvicorn
# Assumes your FastAPI instance is named `app` inside app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
