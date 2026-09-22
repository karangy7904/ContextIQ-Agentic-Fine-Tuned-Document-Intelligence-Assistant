FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-app.txt .

RUN pip install --no-cache-dir -r requirements-app.txt

COPY . .

EXPOSE 7860

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')"

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=7860"]
