FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files the API server needs (not Streamlit app, tests, etc.)
COPY api.py retrieve.py generate.py ingest.py ./

# Render injects env vars at runtime; PORT defaults to 8000
ENV PORT=8000
EXPOSE ${PORT}

# Use uvicorn directly — Render expects the process on $PORT
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}
