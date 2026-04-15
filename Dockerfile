FROM python:3.12-slim

WORKDIR /app

# Copy source and config first so pip install can find the package
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

COPY alembic.ini .
COPY alembic/ alembic/
COPY data/ data/
COPY scripts/ scripts/

EXPOSE 8001

CMD ["uvicorn", "plato_rag.main:app", "--host", "0.0.0.0", "--port", "8001"]
