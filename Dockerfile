# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port (optional, for reference)
EXPOSE 9696

# Use Gunicorn with $PORT assigned by Render
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "predict:app", "--workers", "4"]


