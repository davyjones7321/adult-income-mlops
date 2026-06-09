FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY src/ ./src/
COPY api/ ./api/
COPY mlruns/ ./mlruns/

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]