FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/wheelbot

RUN apt-get update \
 && apt-get install -y --no-install-recommends tini \
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
COPY scripts/ ./scripts/
COPY config/ ./config/

RUN pip install --upgrade pip \
 && pip install -e ".[broker,dashboard,intelligence,data]"

RUN useradd --create-home --uid 1000 wheelbot \
 && mkdir -p /mnt/wheelbot-storage \
 && chown -R wheelbot:wheelbot /opt/wheelbot /mnt/wheelbot-storage

USER wheelbot

EXPOSE 8889

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "strategies.wheel"]
