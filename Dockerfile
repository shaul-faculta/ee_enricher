FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default port Render/Heroku-style
ENV PORT=5000

CMD ["bash", "-lc", "gunicorn -w 1 -k gthread --threads 8 --timeout 120 -b 0.0.0.0:${PORT:-5000} app:app"]


