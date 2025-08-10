FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default port Render/Heroku-style
ENV PORT=5000

CMD ["gunicorn", "-w", "3", "-b", "0.0.0.0:5000", "app:app"]


