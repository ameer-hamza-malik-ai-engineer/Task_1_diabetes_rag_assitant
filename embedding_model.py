from huggingface_hub import InferenceClient
import numpy as np
from langchain_core.embeddings import Embeddings
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
INFERENCE_PROVIDER = "scaleway"

class HuggingFaceInferenceEmbeddings(Embeddings):
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def _process(self, raw):
        arr = np.array(raw)
        if arr.ndim == 3:
            arr = arr[0]        # (1, seq, dim) → (seq, dim)
        if arr.ndim == 2:
            arr = arr[-1]       # last-token (EOS) pooling: (seq, dim) → (dim,)
        return arr.tolist()

    def embed_documents(self, texts):
        return [self._process(self.client.feature_extraction(text, model=self.model)) for text in texts]

    def embed_query(self, text):
        return self._process(self.client.feature_extraction(text, model=self.model))

def get_embedding_model():
    client = InferenceClient(
        provider=INFERENCE_PROVIDER,
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    embedding_model = HuggingFaceInferenceEmbeddings(client=client, model=EMBEDDING_MODEL)

    return embedding_model