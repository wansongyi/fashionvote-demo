FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0 torchvision==0.29.0 && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py inference.py ./
COPY model ./model
COPY static ./static
COPY templates ./templates

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD exec gunicorn --bind :${PORT} --workers 1 --threads 2 --timeout 120 --access-logfile - app:app
