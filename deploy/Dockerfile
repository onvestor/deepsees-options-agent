# Read-only dashboard over completed decision logs. A static demo, not a live
# feed: sessions are baked in at build time and the container never reads a
# broker, a config file, or a prompt.
#
# Built from a staging directory OUTSIDE the repository. That is deliberate:
# the first attempt built from inside it and silently picked up the repo's
# .dockerignore, whose `logs/` deny rule dropped the sessions from a 16 kB
# upload. Staging outside removes every such interaction -- what is here is
# exactly what ships.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only the dashboard package. src/agents, src/brokers, src/options, src/risk
# and src/orchestrator are absent, so no route can construct a broker client
# even if one tried.
COPY src/ ./src/
COPY logs/ ./logs/

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000
# JSON form so signals reach uvicorn; sh -c so Railway's $PORT expands.
CMD ["sh", "-c", "uvicorn src.dashboard.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
