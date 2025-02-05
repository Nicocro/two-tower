# Base image with Python 3.12
FROM python:3.12-slim

# Set working directory - now using two-tower as root
WORKDIR /two-tower

# Install poetry
RUN pip install poetry

# Copy only requirements
COPY pyproject.toml poetry.lock ./

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies (new syntax)
RUN poetry install --no-interaction --no-root

# Copy application code and necessary files maintaining the same structure as local
COPY src/ ./src/
COPY weights/ ./weights/
COPY text8_vocab.json ./
COPY cbow_text8_weights.pt ./

# Expose port
EXPOSE 8000

# Run the application using the exact same command as local
CMD ["poetry", "run", "python", "-m", "src.deploy.app.main"]