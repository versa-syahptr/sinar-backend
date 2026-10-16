# SINAR demo viewer: plays recordings made by pipeline.py, no models inside.
# Build from this directory: docker compose up -d --build
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py records.py logo.svg ./

# clips + *.demo.pkl + *.detect.mp4 / *.track.mp4, mounted by compose
ENV SINAR_DEMO_VID_DIR=/data/vid \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_CLIENT_TOOLBAR_MODE=minimal

RUN useradd --system --uid 10001 --no-create-home sinar
USER sinar

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py"]
