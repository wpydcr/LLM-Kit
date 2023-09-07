import threading
from threading import Thread
from utils.ui_utils import chat_base_model
from queue import Queue
from utils.local_doc import local_doc_qa
import gradio as gr


class api_thread(Thread):
    def __init__(self, params):
        super(api_thread, self).__init__()
        self._stop_event = threading.Event()
        self.chat_model = chat_base_model()
        self.chat_model.load_api_params(params)
        self.inputs = Queue()
        self.outputs = Queue()

    def setv(self, chatbot, api_stream, net, search, search_key, result_len, prompt, doc_qa=None):
        self.chatbot = chatbot
        self.api_stream = api_stream
        self.net = net
        self.search = search
        self.search_key = search_key
        self.result_len = result_len
        self.prompt = prompt
        self.doc_qa = doc_qa

    def run(self):
        while True:
            input = self.inputs.get()
            print(input)
            if input is None:
                break
            try:
                for chatbot, _, _ in self.chat_model.predict(input=input, chatbot=self.chatbot, stream=self.api_stream, net=self.net, search=self.search, search_key=self.search_key, result_len=self.result_len, prompt=self.prompt, doc=self.doc_qa, doc_type='api parallel'):
                    self.outputs.put(chatbot)
                self.outputs.put(None)
            except Exception as e:
                self.chatbot.append([input, 'api thread error'])
                self.outputs.put(self.chatbot)
                break
        self.stop()

    def stop(self):
        if not self.stopped():
            self.inputs.put(None)
            self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class Parallel_api(object):
    def __init__(self):
        self.api_threads = {}

    def setv(self, params):
        self.api_threads = {}
        if 'openai' in params['api_list']:
            api_params = {
                'name': 'openai',
                'api_key': params['openai_api_key'],
                'port': params['openai_port'],
                'api_base': params['openai_api_base'],
                'api_model':params['openai_api_model'],
                'prompt': params['prompt']
            }
            self.api_threads['openai'] = api_thread(api_params)
            if params['use_knowledge']:
                self.api_threads['openai knowledge'] = api_thread(api_params)
        if 'azure openai' in params['api_list']:
            api_params = {
                'name': 'azure openai',
                'api_key': params['azure_api_key'],
                'endpoint': params['azure_endpoint'],
                'engine': params['azure_engine'],
                'prompt': params['prompt']
            }
            self.api_threads['azure openai'] = api_thread(api_params)
            if params['use_knowledge']:
                self.api_threads['azure openai knowledge'] = api_thread(
                    api_params)
        if 'ernie bot' in params['api_list']:
            api_params = {
                'name': 'ernie bot',
                'api_key': params['ernie_api_key'],
                'secret_key': params['ernie_secret_key'],
                'temperature': params['ernie_temperature'],
                'top_p': params['ernie_top_p'],
                'penalty_score': params['ernie_penalty_score']
            }
            self.api_threads['ernie bot'] = api_thread(api_params)
            if params['use_knowledge']:
                self.api_threads['ernie bot knowledge'] = api_thread(
                    api_params)
        if 'ernie bot turbo' in params['api_list']:
            api_params = {
                'name': 'ernie bot turbo',
                'api_key': params['ernie_turbo_api_key'],
                'secret_key': params['ernie_turbo_secret_key']
            }
            self.api_threads['ernie bot turbo'] = api_thread(api_params)
            if params['use_knowledge']:
                self.api_threads['ernie bot turbo knowledge'] = api_thread(
                    api_params)
        if 'chatglm api' in params['api_list']:
            api_params = {
                'name': 'chatglm api',
                'api_key': params['chatglm_api_key'],
                'temperature': params['chatglm_temperature'],
                'top_p': params['chatglm_top_p'],
                'type': params['chatglm_type']
            }
            self.api_threads['chatglm api'] = api_thread(api_params)
            if params['use_knowledge']:
                self.api_threads['chatglm api knowledge'] = api_thread(
                    api_params)
        if 'spark api' in params['api_list']:
            api_params = {
                'name': 'spark api',
                'appid': params['spark_appid'],
                'api_key': params['spark_api_key'],
                'secret_key': params['spark_secret_key'],
                "api_version": params['spark_api_version'],
                'temperature': params['spark_temperature'],
                'top_k': params['spark_top_k'],
                'max_tokens': params['spark_max_tokens']
            }
            self.api_threads['spark api'] = api_thread(api_params)
            if params['use_knowledge']:
                self.api_threads['spark api knowledge'] = api_thread(
                    api_params)
        if 'ali api' in params['api_list']:
            api_params = {
                'name': 'ali api',
                'api_key': params['ali_api_key'],
                'top_p': params['ali_top_p'],
                'top_k': params['ali_top_k'],
                'kuake_search': params['ali_kuake_search']
            }
            self.api_threads['ali api'] = api_thread(api_params)
            if params['use_knowledge']:
                self.api_threads['ali api knowledge'] = api_thread(api_params)
        if params['use_knowledge']:
            self.doc_qa = local_doc_qa()
            if 'openai' == params['emb_name']:
                doc_params = {
                    'name': params['emb_name'],
                    'api_key': params['emb_api_key'],
                    'port': params['emb_port'],
                    'api_base': params['emb_api_base'],
                    'api_model': params['emb_api_model'],
                    'doc': params['doc'],
                    'k': params['k'],
                    'score_threshold': params['score_threshold'],
                    'chunk_size': params['chunk_size'],
                    'chunk_conent': params['chunk_conent']

                }
            elif 'azure openai' == params['emb_name']:
                doc_params = {
                    'name': params['emb_name'],
                    'api_key': params['emb_api_key'],
                    'endpoint': params['emb_endpoint'],
                    'engine': params['emb_engine'],
                    'doc': params['doc'],
                    'k': params['k'],
                    'score_threshold': params['score_threshold'],
                    'chunk_size': params['chunk_size'],
                    'chunk_conent': params['chunk_conent']

                }
            else:
                doc_params = {
                    'name': params['emb_name'],
                    'doc': params['doc'],
                    'k': params['k'],
                    'score_threshold': params['score_threshold'],
                    'chunk_size': params['chunk_size'],
                    'chunk_conent': params['chunk_conent']
                }
            self.doc_qa.load(doc_params)
        else:
            self.doc_qa = None
        for thread in self.api_threads.values():
            thread.start()
        return True

    def call_openai(self, user_input, chatbot, api_stream, net, search, search_key, result_len, prompt):

        if self.api_threads.get('openai', None) is None:
            yield None, ''
        else:
            self.api_threads['openai'].setv(
                chatbot, api_stream, net, search, search_key, result_len, prompt)
            self.api_threads['openai'].inputs.put(user_input)
            while True:
                chatbot = self.api_threads['openai'].outputs.get()
                if chatbot is None:
                    break
                yield chatbot, ''

    def call_openai_knowledge(self, user_input, chatbot, api_stream, net, search, search_key, result_len, prompt):

        if self.api_threads.get('openai knowledge', None) is None:
            yield None, ''
        else:
            self.api_threads['openai knowledge'].setv(
                chatbot, api_stream, net, search, search_key, result_len, prompt, doc_qa=self.doc_qa)
            self.api_threads['openai knowledge'].inputs.put(user_input)
            while True:
                chatbot = self.api_threads['openai knowledge'].outputs.get()
                if chatbot is None:
                    break
                yield chatbot, ''

    def call_azure(self, user_input, chatbot, api_stream, net, search, search_key, result_len, prompt):

        if self.api_threads.get('azure openai', None) is None:
            yield None, ''
        else:
            self.api_threads['azure openai'].setv(
                chatbot, api_stream, net, search, search_key, result_len, prompt)
            self.api_threads['azure openai'].inputs.put(user_input)
            while True:
                chatbot = self.api_threads['azure openai'].outputs.get()
                if chatbot is None:
                    break
                yield chatbot, ''

    def call_azure_knowledge(self, user_input, chatbot, api_stream, net, search, search_key, result_len, prompt):

        if self.api_threads.get('azure openai knowledge', None) is None:
            yield None, ''
        else:
            self.api_threads['azure openai knowledge'].setv(
                chatbot, api_stream, net, search, search_key, result_len, prompt, doc_qa=self.doc_qa)
            self.api_threads['azure openai knowledge'].inputs.put(user_input)
            while True:
                chatbot = self.api_threads['azure openai knowledge'].outputs.get(
                )
                if chatbot is None:
                    break
                yield chatbot, ''

    def call_ernie(self, user_input, chatbot, api_stream, net, search, search_key, result_len, prompt):

        if self.api_threads.get('ernie bot', None) is None:
            yield None, ''
        else:
            self.api_threads['ernie bot'].setv(
                chatbot, api_stream, net, search, search_key, result_len, prompt)
            self.api_threads['ernie bot'].inputs.put(user_input)
            while True:
                chatbot = self.api_threads['ernie bot'].outputs.get()
                if chatbot is None:
                    break
                yield chatbot, ''

    def call_ernie_knowledge(self, user_input, chatbot, api_stream, net, search, search_key, result_len, prompt):

        if self.api_threads.get('ernie bot knowledge', None) is None:
            yield None, ''
        else:
            self.api_threads['ernie bot knowledge'].setv(
                chatbot, api_stream, net, search, search_key, result_len, prompt, doc_qa=self.doc_qa)
            self.api_threads['ernie bot knowledge'].inputs.put(user_input)
            while True:
                chatbot = self.api_threads['ernie bot knowledge'].outputs.get()
                if chatbot is None:
                    break
                yield chatbot, ''

    def call_ernie_turbo(self, user_input, chatbot, api_stream, net, search, search_key, result_len, prompt):

        if self.api_threads.get('ernie bot turbo', None) is None:
            yield None, ''
        else:
            self.api_threads['ernie bot turbo'].setv(
                chatbot, api_stream, net, search, search_key, result_len, prompt)
            self.api_threads['ernie bot turbo'].inputs.put(user_input)
            while True:
                chatbot = self.api_threads['ernie bot turbo'].outputs.get()
                if chatbot is None:
                    break
                yield chatbot, ''

    def call_ernie_turbo_knowledge(self, user_input, chatbot, api_stream, net, search, search_key, result_len, prompt):

        if self.api_threads.get('ernie bot turbo knowledge', None) is None:
            yield None, ''
        else:
            self.api_threads['ernie bot turbo knowledge'].setv(
                chatbot, api_stream, net, search, search_key, result_len, prompt, doc_qa=self.doc_qa)
            self.api_threads['ernie bot turbo knowledge'].inputs.put(
                user_input)
            while True:
                chatbot = self.api_threads['ernie bot turbo knowledge'].outputs.get(
                )
                if chatbot is None:
                    break
                yield chatbot, ''

    def call_chatglm(self, user_input, chatbot, api_stream, net, search, search_key, result_len, prompt):

        if self.api_threads.get('chatglm api', None) is None:
            yield None, ''
        else:
            self.api_threads['chatglm api'].setv(
                chatbot, api_stream, net, search, search_key, result_len, prompt)
            self.api_threads['chatglm api'].inputs.put(user_input)
            while True:
                chatbot = self.api_threads['chatglm api'].outputs.get()
                if chatbot is None:
                    break
                yield chatbot, ''

    def call_chatglm_kmowledge(self, user_input, chatbot, api_stream, net, search, search_key, result_len, prompt):

        if self.api_threads.get('chatglm api knowledge', None) is None:
            yield None, ''
        else:
            self.api_threads['chatglm api knowledge'].setv(
                chatbot, api_stream, net, search, search_key, result_len, prompt, doc_qa=self.doc_qa)
            self.api_threads['chatglm api knowledge'].inputs.put(user_input)
            while True:
                chatbot = self.api_threads['chatglm api knowledge'].outputs.get(
                )
                if chatbot is None:
                    break
                yield chatbot, ''

    def call_spark(self, user_input, chatbot, api_stream, net, search, search_key, result_len, prompt):

        if self.api_threads.get('spark api', None) is None:
            yield None, ''
        else:
            self.api_threads['spark api'].setv(
                chatbot, api_stream, net, search, search_key, result_len, prompt)
            self.api_threads['spark api'].inputs.put(user_input)
            while True:
                chatbot = self.api_threads['spark api'].outputs.get()
                # print(chatbot)
                if chatbot is None:
                    break
                yield chatbot, ''

    def call_spark_knowledge(self, user_input, chatbot, api_stream, net, search, search_key, result_len, prompt):

        if self.api_threads.get('spark api knowledge', None) is None:
            yield None, ''
        else:
            self.api_threads['spark api knowledge'].setv(
                chatbot, api_stream, net, search, search_key, result_len, prompt, doc_qa=self.doc_qa)
            self.api_threads['spark api knowledge'].inputs.put(user_input)
            while True:
                chatbot = self.api_threads['spark api knowledge'].outputs.get()
                if chatbot is None:
                    break
                yield chatbot, ''

    def call_ali(self, user_input, chatbot, api_stream, net, search, search_key, result_len, prompt):
        if self.api_threads.get('ali api', None) is None:
            yield None, ''
        else:
            self.api_threads['ali api'].setv(
                chatbot, api_stream, net, search, search_key, result_len, prompt)
            self.api_threads['ali api'].inputs.put(user_input)
            while True:
                chatbot = self.api_threads['ali api'].outputs.get()
                if chatbot is None:
                    break
                yield chatbot, ''

    def call_ali_knowledge(self, user_input, chatbot, api_stream, net, search, search_key, result_len, prompt):
        if self.doc_qa is None or self.api_threads.get('ali api knowledge', None) is None:
            yield None, ''
        else:
            self.api_threads['ali api knowledge'].setv(
                chatbot, api_stream, net, search, search_key, result_len, prompt, self.doc_qa)
            self.api_threads['ali api knowledge'].inputs.put(user_input)
            while True:
                chatbot = self.api_threads['ali api knowledge'].outputs.get()
                if chatbot is None:
                    break
                yield chatbot, ''

    def clear_history(self):
        for api_thread in self.api_threads.values():
            api_thread.chat_model.reset_state()
            api_thread.inputs.queue.clear()
            api_thread.outputs.queue.clear()

    def clear(self):
        self.clear_history()
        for api_thread in self.api_threads.values():
            api_thread.stop()


class ParallelLocalModel():
    def __init__(self):
        self.chatbot1_vector_store = None
        self.chatbot2_vector_store = None

    @staticmethod
    def handle_local_model_selected():
        return gr.update(visible=True), gr.update(visible=True)

    def load_embedding_params(self, status, switch_chatbot, doc1, k, score_threshold, chunk_size, chunk_conent, emb_api_list, emb_model_list, *args):
        params = {}
        params['doc'] = doc1
        params['k'] = k
        params['score_threshold'] = score_threshold
        params['chunk_size'] = chunk_size
        params['chunk_conent'] = chunk_conent
        if emb_api_list is not None:
            if emb_api_list == 'openai':
                params['name'] = 'openai'
                params['api_key'] = args[0]
                params['port'] = args[1]
                params['api_base'] = args[2]
                params['api_model'] = args[3]
            elif emb_api_list == 'azure openai':
                params['name'] = 'azure openai'
                params['api_key'] = args[4]
                params['endpoint'] = args[5]
                params['engine'] = args[6]
            else:
                pass
        elif emb_model_list is not None:
            params['name'] = emb_model_list

        if switch_chatbot == "chatbot1":
            if self.chatbot1_vector_store == None:
                self.chatbot1_vector_store = local_doc_qa()
            self.chatbot1_vector_store.load(params)
            current = status.Chatbot2[0]
            return gr.update(value=[[doc1, current]])
        elif switch_chatbot == "chatbot2":
            if self.chatbot2_vector_store == None:
                self.chatbot2_vector_store = local_doc_qa()
            self.chatbot2_vector_store.load(params)
            current = status.Chatbot1[0]
            return gr.update(value=[[current, doc1]])

    def clear(self, switch_chatbot, status):
        if switch_chatbot == "chatbot1":
            self.chatbot1_vector_store = None
            vec_to_kep = status.Chatbot2[0]
            return gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(
                value=[["", vec_to_kep]])
        elif switch_chatbot == "chatbot2":
            self.chatbot2_vector_store = None
            vec_to_kep = status.Chatbot1[0]

            return gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=[[vec_to_kep, ""]])
