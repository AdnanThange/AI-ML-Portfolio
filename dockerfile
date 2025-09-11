# Use lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy only the Python script
COPY heart.py .

# Install Python dependencies
RUN pip install --no-cache-dir pandas scikit-learn matplotlib seaborn prometheus_client

# Expose the port your app uses
EXPOSE 8000

# Run the Python script
CMD ["python", "heart.py"]
