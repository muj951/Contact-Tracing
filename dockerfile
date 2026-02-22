# Use the official Python image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy your entire project into the container (except what's in .dockerignore)
COPY . .

# Install the dependencies
RUN uv sync

# Let Docker train the model internally so MLflow generates safe Linux paths!
RUN uv run python -m src.train

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application when the container starts
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
