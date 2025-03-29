FROM python:3.11-slim-buster

WORKDIR /app

# COPY api_gateway/requirements.txt .

COPY ./api_gateway .

RUN pip install --no-cache-dir -U pymilvus
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /app/uploads

# COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
