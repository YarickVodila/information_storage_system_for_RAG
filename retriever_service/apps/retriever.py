from sentence_transformers import SentenceTransformer
import numpy as np
import torch

class Retriever:
    def __init__(self, model_name:str = "intfloat/multilingual-e5-small"):
        self.model = SentenceTransformer(model_name)
        self.model.eval()

        self.embedding_dim = self.model.get_sentence_embedding_dimension() # получение размерности модели


    def get_embeddings(self, texts:list[str], is_query: bool = True, **kwargs) -> np.array:
        """ 

        """

        with torch.inference_mode():
            if is_query:
                embeddings = self.model.encode(sentences = ["query: "+t for t in texts], normalize_embeddings=True, **kwargs)
            else:
                embeddings = self.model.encode(sentences = ["passage: "+t for t in texts], normalize_embeddings=True, **kwargs)

        return embeddings
    

    
