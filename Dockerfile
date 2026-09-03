FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py inference.py ./
COPY model ./model
COPY static ./static
COPY templates ./templates

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD exec gunicorn --bind :${PORT} --workers 1 --threads 4 --timeout 120 --access-logfile - app:app
