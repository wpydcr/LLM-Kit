import os
from langchain.chat_models import ChatOpenAI,AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ChatMessageHistory
import gradio as gr
import openai
import requests
import json
import zhipuai
from modules.model.SparkApi import Spark_Api
import dashscope
from dashscope import Generation
from http import HTTPStatus
import json


real_path = os.path.split(os.path.realpath(__file__))[0]

def get_access_token(API_Key,Secret_Key):

    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_Key}&client_secret={Secret_Key}"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers)
        return json.loads(response.text)['access_token']
    except:
        return None


class ali_api():
    def __init__(self):
        self.history = ChatMessageHistory()

    def get_embedding(self):
        pass

    def setv(self,api_key,top_p=0.8,top_k=100.0,kuake_search=False):
        if api_key == '':
            raise gr.Error('请输入api_key')
        dashscope.api_key = api_key

    def get_ones_openai(self,message):
        history = []
        if type(message)==str:
            message = message.replace("\n", " ")
            messages = message
        else:
            messages = message[:-1]['content']
            for item in message:
                if item['role']=='user':
                    history.append({
                        'user':item['content']
                    })
                elif item['role']=='assistant':
                    history.append({
                        'bot':item['content']
                    })
        try:
            response=Generation.call(
                model='qwen-v1',
                history=history,
                prompt=messages
                )
            if response.status_code==HTTPStatus.OK:
                return response.output.text
            else:
                return 'Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message)
        except:
            return '网络错误'
        
    def get_ones_openai_stream(self,message):
        history = []
        if type(message)==str:
            message = message.replace("\n", " ")
            messages = message
        else:
            messages = message[:-1]['content']
            for item in message:
                if item['role']=='user':
                    history.append({
                        'user':item['content']
                    })
                elif item['role']=='assistant':
                    history.append({
                        'bot':item['content']
                    })
        try:
            for response in Generation.call(
                model='qwen-v1',
                history=history,
                prompt=messages,
                stream=True
                ):
                yield response.output.text
        except:
            return '网络错误'
        
    def cut_memory(self):
        for _ in range(2):
            '''删除一轮对话'''
            first = self.history.messages.pop(0)
            print(f'删除上下文记忆: {first}')

    def talk(self,message):
        history = []
        if type(message)==str:
            message = message.replace("\n", " ")
            messages = message
        else:
            messages = message[:-1]['content']
            for item in message:
                if item['role']=='user':
                    history.append({
                        'user':item['content']
                    })
                elif item['role']=='assistant':
                    history.append({
                        'bot':item['content']
                    })
        try:
            response=Generation.call(
                model='qwen-v1',
                history=history,
                prompt=messages
                )
            if response.status_code==HTTPStatus.OK:
                reply = response.output.text
                self.history.messages.append(reply)
                return reply
            else:
                return 'Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message)
        except:
            self.history.messages = self.history.messages[:-1]
            return '网络错误'
        
    def clear_history(self):
        self.history = ChatMessageHistory()

class spark_api():
    def __init__(self):
        self.history = ChatMessageHistory()

    def get_embedding(self):
        pass

    def setv(self,spark_api_key=None,spark_api_secret=None,spark_appid=None,temperature=0.95,top_k=4,max_tokens=2048,gpt_url="ws://spark-api.xf-yun.com/v1.1/chat"):
        if spark_api_key == '':
            raise gr.Error('请输入spark_api_key')
        if spark_api_secret == '':
            raise gr.Error('请输入spark_api_secret')
        if spark_appid == '':
            raise gr.Error('请输入spark_appid')
        self.spark_api_key = spark_api_key
        self.spark_api_secret = spark_api_secret
        self.spark_appid = spark_appid
        self.temperature = temperature
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.gpt_url = gpt_url
        self.spark_api = Spark_Api(self.spark_appid,self.spark_api_secret,self.spark_api_key,self.temperature,self.top_k,self.max_tokens,self.gpt_url)
        self.spark_api.create_ws()

    def get_ones_openai(self,message):
        if type(message)==str:
            message = message.replace("\n", " ")
            messages = [{'role':'user','content':message}]
        else:
            messages=message
        try:
            total_text = ''
            for text in self.spark_api._call(messages):
                total_text += text
            return total_text
        except:
            return '网络错误'
        
    def get_ones_openai_stream(self,message):
        if type(message)==str:
            message = message.replace("\n", " ")
            messages = [{'role':'user','content':message}]
        else:
            messages=message
        try:
            for text in self.spark_api._call(messages):
                yield text
        except:
            return '网络错误'
        
    def cut_memory(self):
        for _ in range(2):
            '''删除一轮对话'''
            first = self.history.messages.pop(0)
            print(f'删除上下文记忆: {first}')

    def talk(self,message):
        message = message.replace("\n", " ")
        messages = []
        self.history.messages.append(message)
        for i,message in enumerate(self.history.messages):
            if i % 2 == 0:
                messages.append({'role':'user','content':message})
            else:
                messages.append({'role':'assistant','content':message})
        try:
            total_text = ''
            for text in self.spark_api._call(messages):
                total_text += text
            self.history.messages.append(total_text)
            return total_text
        except:
            self.history.messages = self.history.messages[:-1]
            return '网络错误'
        
    def clear_history(self):
        self.history = ChatMessageHistory()
    

class chatglm_api():
    def __init__(self):
        self.history = ChatMessageHistory()

    def get_embedding(self):
        pass

    def setv(self,api_key=None,temperature=0.95,top_p=0.7,chatglm_type='std'):
        if api_key == '':
            raise gr.Error('请输入api_key')
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.chatglm_type = chatglm_type
        zhipuai.api_key = api_key

    def get_ones_openai(self,message):
        if type(message)==str:
            message = message.replace("\n", " ")
            messages = {'role':'user','content':message}
        else:
            messages=message
        try:
            response = zhipuai.model_api.invoke(
                model='chatglm_'+self.chatglm_type,
                prompt=messages,
                temperature=0.95,
                top_p=0.7)
            return response['data']['choices'][0]['content'][1:-1]
        except:
            return '网络错误'
        
    def get_ones_openai_stream(self,message):
        return self.get_ones_openai(message)
    
    def cut_memory(self):
        for _ in range(2):
            '''删除一轮对话'''
            first = self.history.messages.pop(0)
            print(f'删除上下文记忆: {first}')
    
    def talk(self,message):
        message = message.replace("\n", " ")
        messages = []
        self.history.messages.append(message)
        for i,message in enumerate(self.history.messages):
            if i % 2 == 0:
                messages.append({'role':'user','content':message})
            else:
                messages.append({'role':'assistant','content':message})
        try:
            reply = self.get_ones_openai(messages)
            self.history.messages.append(reply)
            return reply
        except:
            self.history.messages = self.history.messages[:-1]
            return '网络错误'

    def clear_history(self):
        self.history = ChatMessageHistory()

class ernie_api():
    def __init__(self) -> None:
        self.history = ChatMessageHistory()

    def get_embedding(self):
        pass

    def setv(self,ernie_api_key=None,ernie_secret_key=None,ernie_temperature=0.95,ernie_top_p=0.8,ernie_penalty_score=1,ernie_type='ernie bot'):
        if ernie_api_key == '':
            raise gr.Error('请输入ernie_api_key')
        if ernie_secret_key == '':
            raise gr.Error('请输入ernie_secret_key')
        self.ernie_type = ernie_type
        if ernie_type == 'ernie bot':
            self.url = f'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions'
        elif ernie_type == 'ernie bot turbo':
            self.url = f'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant'
        else:
            return None

        self.headers = {
            'Content-Type': 'application/json',
        }

        self.access_token = get_access_token(ernie_api_key,ernie_secret_key)

        self.query = {
            'access_token':self.access_token
        }

        self.temperature = ernie_temperature
        self.top_p = ernie_top_p
        self.penalty_score = ernie_penalty_score

    def get_ones_openai(self,message):
        if type(message)==str:
            message = message.replace("\n", " ")

            if self.ernie_type == 'ernie bot':
                body = {
                    'messages':[
                        {'role':'user','content':message},
                    ],
                    'temperature':self.temperature,
                    'top_p':self.top_p,
                    'penalty_score':self.penalty_score,
                    'stream':True
                }
            elif self.ernie_type == 'ernie bot turbo':
                body = {
                    'messages':[
                        {'role':'user','content':message},
                    ],
                    'stream':False
                }
        else:
            messages=message
            body = {
                'messages':messages,
                'stream':False
            }

        try:
            response = requests.request("POST", self.url, headers=self.headers, params=self.query, data=json.dumps(body))
            return json.loads(response.text)['result']
        except:
            return '网络错误'

    def get_ones_openai_stream(self,message):
        if type(message)==str:
            message = message.replace("\n", " ")

            if self.ernie_type == 'ernie bot':
                body = {
                    'messages':[
                        {'role':'user','content':message},
                    ],
                    'temperature':self.temperature,
                    'top_p':self.top_p,
                    'penalty_score':self.penalty_score,
                    'stream':True
                }
            elif self.ernie_type == 'ernie bot turbo':
                body = {
                    'messages':[
                        {'role':'user','content':message},
                    ],
                    'stream':True
                }
        else:
            messages=message
            body = {
                'messages':messages,
                'stream':True
            }

        try:
            response = requests.request("POST", self.url, headers=self.headers, params=self.query, data=json.dumps(body))
            for section in response.iter_lines(decode_unicode=True):
                yield json.loads(section)['result']
        except:
            return '网络错误'

    def cut_memory(self):
        for _ in range(2):
            '''删除一轮对话'''
            first = self.history.messages.pop(0)
            print(f'删除上下文记忆: {first}')

    def talk(self,message):
        message = message.replace("\n", " ")
        body = {
            'messages':[],
            'temperature':self.temperature,
            'top_p':self.top_p,
            'penalty_score':self.penalty_score,
            'stream':False
        }
        self.history.messages.append(message)
        for i,message in enumerate(self.history.messages):
            if i % 2 == 0:
                body['messages'].append({'role':'user','content':message})
            else:
                body['messages'].append({'role':'assistant','content':message})
        try:
            response = requests.request("POST", self.url, headers=self.headers, params=self.query, data=json.dumps(body))
            reply = json.loads(response.text)['result']
            self.history.messages.append(reply)
            return reply
        except:
            self.history.messages = self.history.messages[:-1]
            return '网络错误'

    def clear_history(self):
        self.history = ChatMessageHistory()


class openai_api():
    def __init__(self):
        self.history = ChatMessageHistory()

    def get_embedding(self,openai_api_key,port,type='openai', endpoint='',engine=''):
        if type=='openai':
            openai.api_type = "open_ai"
            if port != None:
                os.environ['http_proxy'] = 'http://127.0.0.1:'+port
                os.environ["https_proxy"] = "http://127.0.0.1:"+port
            self.embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
        elif type=='azure':
            os.environ['OPENAI_API_BASE'] = endpoint
            os.environ['OPENAI_API_KEY'] = openai_api_key
            os.environ['OPENAI_API_TYPE'] = 'azure'
            # 暂未支持
            self.embedding = OpenAIEmbeddings(openai_api_key=openai_api_key,deployment=engine)


    def setv(self,openai_api_key=None,temperature=0.95,max_tokens=4096,top_p=0.7,openai_prompt='',port=10809,model="gpt-3.5-turbo", type='openai', endpoint='',engine=""):
        if type=='openai':
            openai.api_type = "open_ai"
            if port != None:
                os.environ['http_proxy'] = 'http://127.0.0.1:'+port
                os.environ["https_proxy"] = "http://127.0.0.1:"+port
        elif type=='azure':
            pass
        if openai_api_key == '':
            raise gr.Error('请输入openai_api_key')
        self.charactor_prompt=SystemMessage(content=openai_prompt)
        self.max_token=max_tokens
        if max_tokens==0:
            max_tokens=None
            self.max_token=4096
        if type == 'openai':
            self.llm = ChatOpenAI(
                openai_api_key=openai_api_key,
                model_name=model,
                streaming=True,
                temperature=temperature,
                max_tokens=max_tokens)
            self.llm_nonstream = ChatOpenAI(
                openai_api_key=openai_api_key,
                model_name=model,
                streaming=False,
                temperature=temperature,
                max_tokens=max_tokens)
            self.embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
        elif type == 'azure':
            self.llm = AzureChatOpenAI(
                openai_api_key=openai_api_key,
                openai_api_base=endpoint,
                openai_api_type='azure',
                openai_api_version='2023-05-15',
                model_name=model,
                deployment_name=engine,
                streaming=True,
                temperature=temperature,
                top_p=top_p,
                max_tokens=self.max_token)
            self.llm_nonstream = AzureChatOpenAI(
                openai_api_key=openai_api_key,
                openai_api_base=endpoint,
                openai_api_type='azure',
                openai_api_version='2023-05-15',
                model_name=model,
                deployment_name=engine,
                streaming=False,
                temperature=temperature,
                top_p=top_p,
                max_tokens=self.max_token)
            # 暂未支持
            self.embedding = OpenAIEmbeddings(openai_api_key=openai_api_key,deployment=engine)

            
    def get_ones_openai(self,text):
        if type(text)==str:
            text = text.replace("\n", " ")
            message=HumanMessage(content=text)
            messages=[]
            if self.charactor_prompt.content!='':
                messages = [self.charactor_prompt]
            messages.append(message)
        else:
            messages=text
        # if self.llm.get_num_tokens_from_messages(messages)>=self.max_token-500:
        #     return '字数超限'
        return self.llm_nonstream(messages).content
    
    def get_ones_openai_stream(self,text):
        if type(text)==str:
            text = text.replace("\n", " ")
            message=HumanMessage(content=text)
            messages=[]
            if self.charactor_prompt.content!='':
                messages = [self.charactor_prompt]
            messages.append(message)
        else:
            messages=text
        if self.llm.get_num_tokens_from_messages(messages)>=self.max_token-500:
            return '字数超限'
        yield self.llm(messages).content
    
    def cut_memory(self):
        for _ in range(2):
            '''删除一轮对话'''
            first = self.history.messages.pop(0)
            print(f'删除上下文记忆: {first}')

    def talk(self,text):
        text = text.replace("\n", " ")
        message=HumanMessage(content=text)
        messages=[]
        if self.charactor_prompt.content!='':
            messages = [self.charactor_prompt]
        self.history.messages.append(message)
        messages.extend(self.history.messages)
        if self.llm.get_num_tokens_from_messages(messages)>=self.max_token-500:
            self.cut_memory()
        reply=self.llm(messages).content
        self.history.messages.append(AIMessage(content=reply))
        return reply
    
    def clear_history(self):
        self.history = ChatMessageHistory()

