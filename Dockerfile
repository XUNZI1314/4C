FROM python:3.11-slim

WORKDIR /app

# runtime and Streamlit environment optimizations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false

# copy only dependency files first to leverage Docker layer cache
COPY pyproject.toml README.md requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# copy application code after installing deps (cache-friendly)
COPY src ./src
COPY app.py ./
COPY .streamlit ./.streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
