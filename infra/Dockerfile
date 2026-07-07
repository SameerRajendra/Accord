# Accord API image. Fleshed out in Phase 4 (serve + deploy).
# Kept minimal now so the container path is real from the start.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Phase 4 will expose the FastAPI app:
# EXPOSE 8080
# CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["python", "-c", "print('Accord image — API entrypoint arrives in Phase 4')"]
