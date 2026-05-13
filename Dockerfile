FROM python:3.11-slim

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifest first — this layer is cached until pyproject.toml changes
COPY pyproject.toml ./pyproject.toml

# Install runtime dependencies.
# Uses Python 3.11+ built-in tomllib to read deps from pyproject.toml — no requirements.txt needed.
RUN python3 -c "import tomllib,subprocess; d=tomllib.load(open('pyproject.toml','rb')); subprocess.check_call(['uv','pip','install','--system','--no-cache']+d['project']['dependencies'])"

# Copy application source
COPY *.py ./
COPY api/      ./api/
COPY services/ ./services/
COPY handlers/ ./handlers/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "11434"]
