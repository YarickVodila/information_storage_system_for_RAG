from apps.retriever import Retriever

from typing import List
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
import uvicorn
import logging


MODEL_CHECKPOINT = "intfloat/multilingual-e5-small"


retriever = Retriever(model_name = MODEL_CHECKPOINT)

app = FastAPI()

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

class TextRequest(BaseModel):
    texts: List[str]
    is_query: bool = True


@app.post("/create_embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: TextRequest): # texts: list[str], is_query: bool = True
    texts = request.texts
    is_query = request.is_query

    print("Количество текстов: ",len(texts))
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
    uvicorn.run("main:app", port=8000, log_level='info') # , log_level='info', reload=True
