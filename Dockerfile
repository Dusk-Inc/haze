FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /haze

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install -e .[dev]

COPY . .

RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /haze
USER appuser

CMD ["python", "main.py"]
