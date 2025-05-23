FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -v || { echo "pip install failed"; exit 1; } && \
    pip show gunicorn || { echo "gunicorn not installed"; exit 1; }

COPY . .

EXPOSE 8080

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "--timeout", "600", "--log-level", "debug", "--workers", "2", "main:app"]