FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV PYTHONPATH=/app/src

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY configs configs/
COPY src src/


RUN uv sync --frozen --no-cache


ENTRYPOINT ["uv", "run", "-m", "clickbait_classifier.train"]
