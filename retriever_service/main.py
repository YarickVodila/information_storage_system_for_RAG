from apps.retriever import Retriever

from typing import List
import datetime
from pydantic import BaseModel

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import  HTTPBearer
from pymilvus import MilvusClient
import uvicorn
# from requests import request
import logging


MODEL_CHECKPOINT = "intfloat/multilingual-e5-small"


retriever = Retriever(model_name = MODEL_CHECKPOINT)

app = FastAPI()

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

@app.post("/get_embeddings", response_model=EmbeddingResponse)
async def get_embeddings(texts: list[str], is_query: bool = True):
    try:
        if not texts:
            raise HTTPException(status_code=400, detail="Получен пустой список текстов")
        
        embeddings = retriever.get_embeddings(texts, is_query)

        return {"embeddings": embeddings.tolist()}
    
    except Exception as e:
        logging.error(f"Ошибка при генерации эмбеддингов: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")


@app.get("/get_embeddings_dims")
async def get_embeddings_dims():
    embeddings_dims = retriever.embedding_dim
    return {"embeddings_dims": embeddings_dims}


if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8000, log_level='info', reload=True) # , log_level='info', reload=True
