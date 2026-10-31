# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY triage triage
COPY evals evals
RUN pip install --no-cache-dir --prefix=/install ".[mcp]"

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /install /usr/local
COPY examples examples
USER app
ENV TRIAGE_PIPELINE=/app/examples/pipeline.yaml TRIAGE_CASES=/app/examples/case_library.yaml
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
