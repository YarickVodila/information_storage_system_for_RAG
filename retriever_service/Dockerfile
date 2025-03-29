FROM python:3.11-slim-buster

WORKDIR /app

# COPY requirements.txt .
COPY ./retriever_service .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers==4.48.3 && \
    pip install --no-cache-dir huggingface_hub==0.25.2 && \
    pip install --no-cache-dir accelerate>=0.26.0 && \
    pip install --no-cache-dir sentence_transformers==4.0.1

RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0",  "--port", "8000"] 
