from typing import List
import os
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, File, UploadFile
import uvicorn
import httpx
from pymilvus import MilvusClient, DataType
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import numpy as np

DATABASE_SERVICE_URL = "http://localhost:19530"
RETRIEVER_SERVICE_URL = "http://localhost:8000"
UPLOAD_DIR = "uploads"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 400


app = FastAPI()

milvus_client = MilvusClient(DATABASE_SERVICE_URL)

class SearchItem(BaseModel):
    """ 
    Represents a single search result item.

    Fields:
        id (int): Unique identifier of the document
        distance (float): Similarity score between query and document
        entity (dict): Contains the document text and other fields
    """
    id: int
    distance: float
    entity: dict

class SearchRequest(BaseModel):
    """ 
    Request model for search endpoint.

    Fields:
        collection_name (str): Name of the collection to search in
        query (str): Search query text
        top_k (int, optional): Number of results to return (default: 3)
    """
    collection_name: str
    query: str
    top_k: int = 3


@app.post("/create_collection")
async def create_collection(collection_name:str, metric_type:str = "COSINE"):
    """ 
    Creates a new Milvus collection for vector storage.

    Args:
        collection_name (str): Name of the collection to create
        metric_type (str, optional): Similarity metric type ("COSINE", "L2", etc.). Default: "COSINE". More https://milvus.io/docs/ru/metric.md

    Returns:
        dict: Status message

    Raises:
        HTTPException(500): If collection creation fails
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{RETRIEVER_SERVICE_URL}/get_embeddings_dims")

        response = response.json()
        # print(response)

        if milvus_client.has_collection(collection_name=collection_name):
            return {"Status": f"A collection named '{collection_name}' has been created"}

        else:
            schema = MilvusClient.create_schema()

            schema.add_field(
                field_name="id",
                datatype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            )

            schema.add_field(
                field_name="vector", 
                datatype=DataType.FLOAT_VECTOR, 
                dim=response.get("embeddings_dims")
            )

            schema.add_field(
                field_name="text", 
                datatype=DataType.VARCHAR, 
                max_length=2500,
            )

            index_params = milvus_client.prepare_index_params()

            index_params.add_index(
                field_name="vector", 
                index_type="AUTOINDEX",
                metric_type = metric_type
            )

            milvus_client.create_collection(
                collection_name = collection_name,
                # auto_id=False,
                schema = schema,
                index_params = index_params,
                enable_dynamic_field = True,
                # dimension=384
            )



            # milvus_client.create_collection(
            #     collection_name = collection_name,
            #     dimension = response.get("embeddings_dims"),
            #     metric_type = metric_type,
            #     # auto_id = True
            # )
            return {"Status": f"Successful collection creation"}

    except Exception as e:
        logging.error(f"Ошибка при создании коллекции: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")


@app.post("/drop_collection")
async def drop_collection(collection_name:str):
    """ 
    Deletes an existing Milvus collection.

    Args:
        collection_name (str): Name of the collection to delete

    Returns:
        dict: Status message

    Raises:
        HTTPException(400): If collection doesn't exist
        HTTPException(500): If deletion fails
    
    """
    try:
        if milvus_client.has_collection(collection_name=collection_name):
            milvus_client.drop_collection(collection_name=collection_name)
            return {"Status": f"A collection named '{collection_name}' has been deleted"}

        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Коллекции с именем '{collection_name}' нет"
            )

    except Exception as e:
        logging.error(f"Ошибка при удалении коллекции: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")



@app.post("/insert_data_collection")
async def insert_data_collection(collection_name:str, file: UploadFile = File(...)):
    """ 
    Processes a PDF file and inserts its contents into a collection.

    Args:
        collection_name (str): Target collection name
        file (UploadFile): PDF file to process

    Returns:
        dict: Status message

    Raises:
        HTTPException(400): For non-PDF files or missing collection
        HTTPException(500): For processing errors
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Разрешена загрузка только PDF файлов"
        )
    
    filename = file.filename
    file_path = os.path.join(UPLOAD_DIR, filename)
    

    try:
        # Сохраняем файл
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Ошибка при загрузке файла: {str(e)}"
        )
    
    try:
        # Если коллекция есть
        if milvus_client.has_collection(collection_name=collection_name):
            # Разбиваем PDF на чанки
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                is_separator_regex=False,
            )

            loader = PyPDFLoader(file_path = file_path)
            documents = loader.load_and_split(text_splitter)
            
            # r = [len(doc.page_content) for doc in documents]
            # sorted(r, reverse=True)
            # print("Длина документов: ",r[:10])

            documents = [doc.page_content for doc in documents]

            print("Количество документов", len(documents))

            # Получаем эмбеддинги документов
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                response = await client.post(
                    url = f"{RETRIEVER_SERVICE_URL}/create_embeddings",
                    json={"texts": documents, "is_query": False}
                )

            # print(response.status_code)

            if response.status_code == 200:
                embeddings = response.json()
                embeddings = np.array(embeddings["embeddings"])

                data = [{"text": text, "vector": embeddings[i]} for i, text in enumerate(documents)]
                # print(data[0])
                result = milvus_client.insert(collection_name=collection_name, data=data)

                # return {"Количество документов": len(documents), "embeddings" : list(embeddings.shape)}
                return {"Status": "Данные успешно добавлены"}
            

            else:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Ошибка при получении эмбеддингов: {response.text}"
                )
            

        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Ошибка! Коллекция с именем '{collection_name}' не обнаружена"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Ошибка при добавлении данных в коллекцию {collection_name}: {str(e)}"
        )


@app.post("/search", response_model=List[SearchItem])
async def search(request: SearchRequest):
    """ 
    Performs semantic search on a collection.

    Args:
        request (SearchRequest): Contains collection name, query, and top_k

    Returns:
        List[SearchItem]: Search results with scores and text

    Raises:
        HTTPException(400): If collection doesn't exist
        HTTPException(500): For search errors
    
    """
    collection_name = request.collection_name
    query = request.query
    top_k = request.top_k

    try:
        # Если коллекция есть
        if milvus_client.has_collection(collection_name=collection_name):
            # Получаем эмбеддинги документов
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url = f"{RETRIEVER_SERVICE_URL}/create_embeddings",
                    json={"texts": [query], "is_query": True}
                )
                
            if response.status_code == 200:
                embeddings = response.json()
                embeddings = np.array(embeddings["embeddings"])

                response = milvus_client.search(
                    collection_name=collection_name,  # target collection
                    data=embeddings,  # query vectors
                    limit=top_k,  # number of returned entities
                    output_fields=["text"],  # specifies fields to be returned
                )

                result = []
                for hits in response:
                    for hit in hits:
                        result.append(hit)

                return result
            

            else:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Ошибка при получении эмбеддингов: {response.text}"
                )
            

        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Ошибка! Коллекция с именем '{collection_name}' не обнаружена"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Ошибка  {collection_name}: {str(e)}"
        )



if __name__ == "__main__":
    uvicorn.run("main:app", port=8001, log_level='info') # , reload=True
