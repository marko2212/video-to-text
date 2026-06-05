# syntax=docker/dockerfile:1
#
# Run the app in a container. ffmpeg and all Python deps are bundled, so the
# host needs nothing but Docker.
#   docker compose up --build
# Include the offline Whisper backend (bigger image):
#   INSTALL_LOCAL=true docker compose up --build
FROM python:3.12-slim

# ffmpeg is required for audio extraction / decoding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Bring in the uv binary for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

WORKDIR /app
ENV UV_LINK_MODE=copy

# Install dependencies first (better layer caching). INSTALL_LOCAL=true also
# pulls the heavy offline backend (faster-whisper).
COPY pyproject.toml uv.lock ./
ARG INSTALL_LOCAL=false
RUN if [ "$INSTALL_LOCAL" = "true" ]; then \
        uv sync --frozen --no-dev --extra local; \
    else \
        uv sync --frozen --no-dev; \
    fi

# Copy the application code.
COPY . .

EXPOSE 8501

# Streamlit must bind to 0.0.0.0 to be reachable from the host.
CMD ["uv", "run", "--no-sync", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
