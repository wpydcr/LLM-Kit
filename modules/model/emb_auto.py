import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import torch


real_path = os.path.split(os.path.realpath(__file__))[0]

DEVICE_ = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE_ID = "0" if torch.cuda.is_available() else None
DEVICE = f"{DEVICE_}:{DEVICE_ID}" if DEVICE_ID else DEVICE_


class AutoEmb():
    def __init__(self,embedding_model):
        new_path = os.path.join(real_path, "..","..", "models", "Embedding")
        original_path = os.getcwd()
        os.chdir(new_path)
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model,model_kwargs={'device': DEVICE_},cache_folder=new_path)
        os.chdir(original_path)

    # 余弦相似度
    def cos_sim(self, a, b):
        return a.dot(b) / (torch.norm(a) * torch.norm(b))
    
    # 文字转向量
    def get_embedding(self, text):
        return self.embedding.embed_query(text)


