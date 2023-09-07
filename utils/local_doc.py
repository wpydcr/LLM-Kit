

from modules.model.use_api import *
from modules.model.emb_auto import AutoEmb
import os
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader
from utils.chinese_text_splitter import ChineseTextSplitter
from utils.MyFAISS import MyFAISS
import datetime
import gradio as gr
from modules.agent.chatdb.mysql import MySQLDB

real_path = os.path.split(os.path.realpath(__file__))[0]

class local_doc_qa():
    """Local document-based question answering setup class.
        This class response to the input from the webui page and other behaviours, including load vector store, edit vector
        store content.
        这个类用于响应前端的输入和其他行为，包括加载向量数据库，编辑向量数据库内容。



    """
    def __init__(self):
        self.vs_data=None
        self.vector_store = None
        self.prompt_template = """已知信息：
        {context} 

        根据上述已知信息，简洁和专业的来回答用户的问题。不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""
        self.mysql = MySQLDB()
        self.mysql_model = None
        
    def clear(self):
        self.vector_store = None
        return gr.update(value=None),gr.update(value=None),gr.update(value=None)

    def set_v(self,params):
        """Sets the embedding model for vectorization.

        Args:
            embedding_model (str): The name of the embedding model.
            openai_api_key (str): The OpenAI API key.
            port (int): The port number.
            endpoint (str, optional): The endpoint URL. Defaults to 'https://gavency.openai.azure.com/'.
            engine (str, optional): The name of the engine. Defaults to 'chatgpt'.

        """
        if params['name'] == 'openai':
            self.embeddings=openai_api()
            self.embeddings.get_embedding(openai_api_key=params['api_key'],port=params['port'], api_base=params['api_base'], api_model=params['api_model'])
        elif params['name'] == 'azure openai':
            self.embeddings=openai_api()
            self.embeddings.get_embedding(openai_api_key=params['api_key'],port=params['port'],type='azure',endpoint=params['endpoint'],engine=params['engine'])
        else:
            self.embeddings=AutoEmb(params['name'])

    def generate_prompt(self,related_docs,query) -> str:
        context = "\n".join([doc.page_content for doc in related_docs])
        prompt = self.prompt_template.replace("{question}", query).replace("{context}", context)
        return prompt

    def load(self,params):
        """Loads the vector store.

        Args:
            vs_name (str): The name of the vector store.
            emb0 (str): The name of the embedding model.
            k (int): The number of similar documents to retrieve.
            score_threshold (int): The score threshold for relevance.
            chunk_size (int): The chunk size for vectorization.
            chunk_conent (bool): Flag to enable chunk content association.
            openai_api_key (str): The OpenAI API key.
            port (int): The port number.

        """
        if params.get('name',None) == None:
            raise gr.Error("请先选择嵌入式模型")
        if params.get('doc',None) == None:
            raise gr.Error("请先选择知识库")
        self.set_v(params)
        self.vector_store = MyFAISS.load_local(os.path.join(real_path,"..","data", "documents",params['doc']), self.embeddings.embedding)
        self.vector_store.chunk_size = params['chunk_size']
        # chunk_conent   是否启用上下文关联
        self.vector_store.chunk_conent = params['chunk_conent']
        # 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
        self.vector_store.score_threshold = params['score_threshold']
        self.top_k=params['k']

        doc = params['doc']
        print(f'知识库{doc}已成功加载')
        return True

    def get_similarity(self,query):
        related_docs_with_score = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        prompt = self.generate_prompt(related_docs_with_score, query)
        return prompt,related_docs_with_score

    def load_file(self,filepath, sentence_size=100):

        if filepath.lower().endswith(".md"):
            loader = UnstructuredFileLoader(filepath, mode="elements")
            docs = loader.load()
        elif filepath.lower().endswith(".pdf"):
            loader = UnstructuredFileLoader(filepath)
            textsplitter = ChineseTextSplitter(pdf=True)
            docs = loader.load_and_split(textsplitter)
        elif filepath.lower().endswith(".txt"):
            loader = TextLoader(filepath, encoding='utf-8')
            textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
            docs = loader.load_and_split(textsplitter)
        elif filepath.lower().endswith(".csv"):
            loader = CSVLoader(filepath)
            docs = loader.load()
        else:
            loader = UnstructuredFileLoader(filepath, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
            docs = loader.load_and_split(text_splitter=textsplitter)
        return docs

    def upload_data(self,emb_api_list,emb_model_list,files,vs_name,openai_api_key,openai_port,openai_api_base, openai_api_model, azure_api_key,azure_endpoint,azure_engine):
        if emb_api_list is not None:
            if emb_api_list == 'openai':
                params = {
                    'name': emb_api_list,
                    'api_key': openai_api_key,
                    'port': openai_port,
                    'api_base':openai_api_base,
                    'api_model':openai_api_model
                }
            elif emb_api_list == 'azure openai':
                params = {
                    'name': emb_api_list,
                    'api_key': azure_api_key,
                    'endpoint': azure_endpoint,
                    'engine': azure_engine
                }
            else:
                pass
        elif emb_model_list is not None:
            params = {
                'name': emb_model_list
            }
        else:
            raise gr.Error("请先选择嵌入式模型")
        
        self.set_v(params=params)
        loaded_files = ''
        docs = []
        for file in files:
            file=file.name
            try:
                docs += self.load_file(file)
                loaded_files+=os.path.split(file)[-1]+'已成功加载\n'
            except Exception as e:
                print(e)
                loaded_files+=os.path.split(file)[-1]+'未能成功加载\n'
        if len(docs) > 0:
            if not vs_name:
                name = params['name']
                vs_name = f'{name}_FAISS_{datetime.datetime.now().strftime("%m%d_%H%M%S")}'
            vs_path = os.path.join(real_path,"..","data", "documents",vs_name)
            vector_store = MyFAISS.from_documents(docs, self.embeddings.embedding)
            vector_store.save_local(vs_path)
            return loaded_files+'知识库名：'+vs_name
        else:
            return loaded_files+"文件均未成功加载，请检查依赖包或替换为其他文件再次上传。"


    def handle_database_selected(self,create_emb_api_list,create_emb_model_list,docs,create_openai_api_key,create_openai_port,create_openai_api_base, create_openai_api_model, create_azure_api_key,create_azure_endpoint,create_azure_engine):
        """Handles the selection of a vector database on the webui data page. Load the selected embedded model and return
            the files in the database.

        Args:
            doc_name (str): The name of the selected document.
            emb (str): The name of the embedding model.
            openai_api_key (str): The OpenAI API key.
            port (int): The port number.

        Returns:
            str: documents in the vector store.

        """
        real_path = os.path.split(os.path.realpath(__file__))[0]
        if create_emb_api_list is not None:
            if create_emb_model_list == 'openai':
                params = {
                    'name': create_emb_api_list,
                    'api_key': create_openai_api_key,
                    'port': create_openai_port,
                    'api_base':create_openai_api_base,
                    'api_model':create_openai_api_model
                }
            elif create_emb_model_list == 'azure openai':
                params = {
                    'name': create_emb_api_list,
                    'api_key': create_azure_api_key,
                    'endpoint': create_azure_endpoint,
                    'engine': create_azure_engine
                }
            else:
                pass
        elif create_emb_model_list is not None:
            params = {
                'name': create_emb_model_list
            }
        else:
            raise gr.Error("请先选择嵌入式模型")
        if docs is None:
            raise gr.Error("请先选择知识库")
        
        self.set_v(params=params)
        self.vector_store=MyFAISS.load_local(os.path.join(real_path,"..","data", "documents",docs), self.embeddings.embedding)
        return gr.update(choices = self.vector_store.list_docs(), interactive = True)

    def handle_vector_database_file_delete(self,files,database):
        """Handles the deletion of files from a vector store database on the webui data page.

        Args:
            files (List[str]): The list of files to delete.
            database (str): The name of the database.

        Returns:
            str: The updated choices for the vector store documents.

        """
        self.vector_store.delete_doc(files,database)
        return gr.update(choices = self.vector_store.list_docs(), interactive = True)

    def handle_add_file_to_vector_database(self,files,database_name):
        """Handles the addition of files to a vector store database.

               Args:
                   files (List[file]): The list of files to add.
                   database_name (str): The name of the database.

               Returns:
                   str: The updated choices for the vector store documents.

               """
        docs = []
        for file in files:
            docs += self.load_file(file.name)
        self.vector_store.add_files(docs,database_name)
        return gr.update(choices=self.vector_store.list_docs(), interactive=True)

    def get_directories(self,path, unuse):
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d not in unuse]

    def refresh_emb(self):
        real_path = os.path.split(os.path.realpath(__file__))[0]
        new_path = os.path.join(real_path, "..", "models", "Embedding")
        embs = self.get_directories(new_path, [])
        return gr.update(choices = embs)

    def refresh_vector(self):

        """Refreshes the list of vector stores and embedding models.

        Returns:
            List[str]: The updated choices for the vector stores and embedding models.

        """

        real_path = os.path.split(os.path.realpath(__file__))[0]
        new_path = os.path.join(real_path, "..", "data", "documents")
        vector = self.get_directories(new_path, [])
        new_path = os.path.join(real_path, "..", "models", "Embedding")
        embs = self.get_directories(new_path, [])

        return gr.update(choices=vector),gr.update(choices=embs)

    def clear_mysql(self):
        self.sql_use_model = None
        return [],[],'',gr.update(value=None),gr.update(value=None),gr.update(visible=False)