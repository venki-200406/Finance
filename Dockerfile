FROM python:3.12-slim

WORKDIR /srv

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/__init__.py src/predict.py src/
COPY api/ api/
COPY models/classifier.joblib models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
