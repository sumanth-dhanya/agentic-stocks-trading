FROM python:3.12-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory.
WORKDIR /app

# Install the application dependencies.
COPY uv.lock pyproject.toml README.md ./
RUN uv sync --frozen --no-cache

# Copy the application into the container.
COPY src/agentic_stocks_trading agentic_stocks_trading/

CMD ["/app/.venv/bin/fastapi", "run", "agentic_stocks_trading/infrastructure/api/main.py", "--port", "8000", "--host", "0.0.0.0"]

