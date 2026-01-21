FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# deps système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Modèle spaCy AU BUILD (pas au runtime)
RUN python -m spacy download fr_core_news_md

COPY app.py .

EXPOSE 8000

# Uvicorn OK, mais Gunicorn est plus "prod" (reco ici)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app:app", \
     "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "1", \
     "--timeout", "60", "--keep-alive", "5"]
