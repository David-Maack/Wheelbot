FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/wheelbot

RUN apt-get update \
 && apt-get install -y --no-install-recommends tini curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY core/ ./core/
COPY platforms/ ./platforms/
COPY data/ ./data/
COPY strategies/ ./strategies/
COPY execution/ ./execution/
COPY intelligence/ ./intelligence/
COPY risk/ ./risk/
COPY dashboard/ ./dashboard/
COPY db/ ./db/
COPY mcp_server/ ./mcp_server/
COPY scripts/ ./scripts/
COPY config/ ./config/

RUN pip install --upgrade pip \
 && pip install -e ".[broker,dashboard,intelligence,data,mcp]"

RUN useradd --create-home --uid 1000 wheelbot \
 && mkdir -p /mnt/wheelbot-storage \
 && chown -R wheelbot:wheelbot /opt/wheelbot /mnt/wheelbot-storage

USER wheelbot

EXPOSE 8889

ENTRYPOINT ["tini", "--"]
# Default CMD runs the bot. The dashboard is a separate compose service that
# overrides CMD with `uvicorn dashboard.app:create_app --factory ...`.
CMD ["python", "-m", "scripts.run_bot"]
