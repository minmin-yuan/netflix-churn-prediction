# Use lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port Flask/Gunicorn will run on
EXPOSE 9696

# Start Gunicorn server for production
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app", "--workers", "4"]

