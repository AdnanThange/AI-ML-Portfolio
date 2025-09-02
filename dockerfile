# ------------------------------
# Base Image
# ------------------------------
FROM python:3.11-slim

# ------------------------------
# Environment Variables
# ------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH

# ------------------------------
# Set working directory
# ------------------------------
WORKDIR /app

# ------------------------------
# Install dependencies
# ------------------------------
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# ------------------------------
# Copy ML project folder
# ------------------------------
COPY ["Ml and Dl Projects with Mlops/dl_projects_lstm_with_mlflow", "/app/dl_projects_lstm_with_mlflow"]

# ------------------------------
# Set working directory to the project folder
# ------------------------------
WORKDIR /app/dl_projects_lstm_with_mlflow

# ------------------------------
# Default command
# ------------------------------
CMD ["python", "train_imdb.py"]
