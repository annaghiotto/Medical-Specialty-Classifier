FROM python:3.11-slim
WORKDIR /app

COPY requirements-ui.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements-ui.txt

COPY src/ ./src/

EXPOSE 8501
CMD ["streamlit","run","src/app_streamlit.py","--server.address=0.0.0.0","--server.port=8501"]