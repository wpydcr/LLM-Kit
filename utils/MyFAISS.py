import os

from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.faiss import dependable_faiss_import
from typing import Any, Callable, List, Dict
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
import numpy as np

#本模块使用https://github.com/imClumsyPanda/langchain-ChatGLM项目源码

class MyFAISS(FAISS, VectorStore):
    """Customized implementation of FAISS vector store.
        This class implemented the vector-context matching, vectore sotre editing.
        这个类实现了向量数据库的查找以及编辑。
        """
    def __init__(
            self,
            embedding_function: Callable,
            index: Any,
            docstore: Docstore,
            index_to_docstore_id: Dict[int, str],
    ):
        super().__init__(embedding_function=embedding_function,
                         index=index,
                         docstore=docstore,
                         index_to_docstore_id=index_to_docstore_id)

    def seperate_list(self, ls: List[int]) -> List[List[int]]:
        lists = []
        ls1 = [ls[0]]
        for i in range(1, len(ls)):
            if ls[i - 1] + 1 == ls[i]:
                ls1.append(ls[i])
            else:
                lists.append(ls1)
                ls1 = [ls[i]]
        lists.append(ls1)
        return lists

    def similarity_search_with_score_by_vector(
            self, embedding: List[float], k: int = 4
    ) -> List[Document]:
        vector = np.array([embedding], dtype=np.float32)
        scores, indices = self.index.search(vector, k)
        docs = []
        id_set = set()
        store_len = len(self.index_to_docstore_id)
        for j, i in enumerate(indices[0]):
            if i == -1 or 0 < self.score_threshold < scores[0][j]:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if (not self.chunk_conent) or ("context_expand" in doc.metadata and not doc.metadata["context_expand"]):
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                doc.metadata["score"] = int(scores[0][j])
                docs.append(doc)
                continue
            id_set.add(i)
            docs_len = len(doc.page_content)
            for k in range(1, max(i, store_len - i)):
                break_flag = False
                if "context_expand_method" in doc.metadata and doc.metadata["context_expand_method"] == "forward":
                    expand_range = [i + k]
                elif "context_expand_method" in doc.metadata and doc.metadata["context_expand_method"] == "backward":
                    expand_range = [i - k]
                else:
                    expand_range = [i + k, i - k]
                for l in expand_range:
                    if l not in id_set and 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        if docs_len + len(doc0.page_content) > self.chunk_size or doc0.metadata["source"] != doc.metadata["source"]:
                            break_flag = True
                            break
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
                if break_flag:
                    break
        if (not self.chunk_conent) or ("add_context" in doc.metadata and not doc.metadata["add_context"]):
            return docs
        if len(id_set) == 0 and self.score_threshold > 0:
            return []
        id_list = sorted(list(id_set))
        id_lists = self.seperate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += " " + doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
            doc.metadata["score"] = int(doc_score)
            docs.append(doc)
        return docs

    def delete_doc(self, source: str or List[str],docs):
        """Deletes documents from the vector store.

        Args:
            source (str or List[str]): The source(s) of the document(s) to delete.
            docs: The name of the vector store.

        Returns:
            str: The result message.

        """
        real_path = os.path.split(os.path.realpath(__file__))[0]
        vs_path = os.path.join(real_path, "..","data","documents",docs)

        files = list(set(v.metadata["source"] for v in self.docstore._dict.values()))
        files_to_be_delete = []
        for i in source:
            for j in files:
                if i == os.path.split(j)[-1]:
                    files_to_be_delete.append(j)
        try:
            if isinstance(files_to_be_delete, str):
                ids = [k for k, v in self.docstore._dict.items() if v.metadata["source"] == files_to_be_delete]
                # vs_path = os.path.join(real_path, "..","data","documents",docs)
            else:
                ids = [k for k, v in self.docstore._dict.items() if v.metadata["source"] in files_to_be_delete]
                # vs_path = os.path.join(os.path.split(os.path.split(source[0])[0])[0], "vector_store")
            if len(ids) == 0:
                return f"docs delete fail"
            else:
                for id in ids:
                    index = list(self.index_to_docstore_id.keys())[list(self.index_to_docstore_id.values()).index(id)]
                    self.index_to_docstore_id.pop(index)
                    self.docstore._dict.pop(id)
                # self.index.reset()

                self.save_local(vs_path)
                return f"docs delete success"
        except Exception as e:
            print(e)
            return f"docs delete fail"


    def list_docs(self):
        """Returns the list of documents in the vector store.

        Returns:
            List[str]: The list of document names, without path prefix.

        """
        files = list(set(v.metadata["source"] for v in self.docstore._dict.values()))
        for i in range(len(files)):
            files[i] = os.path.split(files[i])[-1]
        return files

    def add_files(self,files,database_name):
        real_path = os.path.split(os.path.realpath(__file__))[0]
        vs_path = os.path.join(real_path, "..", "data", "documents", database_name)
        self.add_documents(files)
        self.save_local(vs_path)


