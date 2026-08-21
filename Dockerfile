FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .
COPY streamlit_app_tfidf.py .
COPY main.py .
COPY models_config.py .
COPY router.py .

COPY DOCS/ ./_data/

EXPOSE 10000

CMD ["sh", "-c", "streamlit run streamlit_app_tfidf.py --server.port=${PORT:-10000} --server.address=0.0.0.0"]
