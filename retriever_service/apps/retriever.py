from sentence_transformers import SentenceTransformer
import numpy as np
import torch

class Retriever:
    """
    A class for generating text embeddings using SentenceTransformer models.
    """

    def __init__(self, model_name:str = "intfloat/multilingual-e5-small"):
        """
        Initializes the Retriever with a specified SentenceTransformer model.

        Args:
            model_name (str, optional): Name/path of the SentenceTransformer model from Hugging Face. Defaults to "intfloat/multilingual-e5-small".
        """
        self.model = SentenceTransformer(model_name, device='cpu')
        self.model.eval()

        self.embedding_dim = self.model.get_sentence_embedding_dimension() # получение размерности модели


    def init_custom_model(self, model_checkpoint: str):
        """
        Reinitializes the Retriever with a custom model checkpoint.

        Args:
            model_checkpoint (str): Name/path of the model checkpoint to initialize from Hugging Face.
        """

        self.model = SentenceTransformer(model_checkpoint, device='cpu')
        self.model.eval()

        self.embedding_dim = self.model.get_sentence_embedding_dimension() # получение размерности модели


    def get_embeddings(self, texts:list[str], is_query: bool = True, **kwargs) -> np.array:
        """ 
        Generates embeddings for the provided texts.

        Args:
            texts (list[str]): List of text strings to generate embeddings for.
            is_query (bool, optional): If True, prepends "query: " to each text; if False, prepends "passage: ". Defaults to True.
            **kwargs: Additional arguments to pass to the model's encode method.

        Returns:
            np.array: Numpy array containing the generated embeddings.
        """

        with torch.inference_mode():
            if is_query:
                embeddings = self.model.encode(sentences = ["query: "+t for t in texts], normalize_embeddings=True, **kwargs)
            else:
                embeddings = self.model.encode(sentences = ["passage: "+t for t in texts], normalize_embeddings=True, **kwargs)

        return embeddings
    
    
    

    
