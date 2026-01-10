FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY solution.py .
COPY helpers.py .

RUN mkdir -p /app/output

ENV PYTHONUNBUFFERED=1

CMD ["python", "solution.py", "--test", "/app/test.csv", "--novels", "/app/novels", "--output", "/app/output/submission.csv"]