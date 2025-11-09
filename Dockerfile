FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9696

# Use CMD as a string so $PORT expands at runtime
CMD gunicorn --bind 0.0.0.0:$PORT predict:app --workers 4
