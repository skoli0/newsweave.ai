# NewsWeave API + agents (Ollama runs in a separate container)
# `podman build -f Containerfile` - kept in sync with Dockerfile
FROM docker.io/library/python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py paths.py orchestrator.py news_agent.py analyst_agent.py graph_agent.py storage_agent.py agentic_workflow.py ./
COPY dashboard.html .

RUN useradd -r -m -d /app -s /bin/false appuser && mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
