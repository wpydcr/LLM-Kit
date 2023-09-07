import os
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
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
from dashscope import Generation, TextEmbedding
from http import HTTPStatus
import json
from modules.model.prompt_generator import prompt_generator

p_generator = prompt_generator()

real_path = os.path.split(os.path.realpath(__file__))[0]


def get_ernie_access_token(API_Key, Secret_Key):

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

    def setv(self, api_key, top_p=0.8, top_k=100.0, kuake_search=False):
        if api_key == '':
            return {
                'status': -1,
                'message': '请输入ali api key'
            }
        dashscope.api_key = api_key
        self.top_p = top_p
        self.top_k = top_k
        self.kuake_search = kuake_search
        return {
            'status': 0,
            'message': '设置成功'
        }

    def get_ones(self, message):
        if type(message) == str:
            message = message.replace("\n", " ")
            messages, history = p_generator.generate_ali_prompt(
                message=message, history=[])
        else:
            messages = message[-1]['user']
            history = message[:-1]
        try:
            response = Generation.call(
                model='qwen-v1',
                history=history,
                prompt=messages,
                top_p=self.top_p,
                top_k=self.top_k,
                enable_search=self.kuake_search,
                stream=False
            )
            if response.status_code == HTTPStatus.OK:
                return {
                    'status': 0,
                    'message': response.output.text
                }
            else:
                print('Error:Code: %d, status: %s, message: %s' %
                      (response.status_code, response.code, response.message))
                return {
                    'status': -1,
                    'message': 'Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message)
                }
        except:
            print('Error:Code: %d, status: %s, message: %s' %
                  (response.status_code, response.code, response.message))
            return {
                'status': -1,
                'message': 'Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message)
            }

    def get_ones_stream(self, message):
        history = []
        if type(message) == str:
            message = message.replace("\n", " ")
            messages, history = p_generator.generate_ali_prompt(
                message=message, history=[])
        else:
            messages = message[-1]['user']
            history = message[:-1]
        try:
            for response in Generation.call(
                model='qwen-v1',
                history=history,
                prompt=messages,
                top_p=self.top_p,
                top_k=self.top_k,
                enable_search=self.kuake_search,
                stream=True
            ):
                if response.status_code == HTTPStatus.OK:
                    yield {
                        'status': 0,
                        'message': response.output.text
                    }
                else:
                    print('Error:Code: %d, status: %s, message: %s' %
                          (response.status_code, response.code, response.message))
                    yield {
                        'status': -1,
                        'message': 'Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message)
                    }
        except:
            print('Code: %d, status: %s, message: %s' %
                  (response.status_code, response.code, response.message))
            yield {
                'status': -1,
                'message': 'Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message)
            }

    def cut_memory(self):
        if len(self.history.messages) == 0:
            return {
                'status': -1,
                'message': '裁剪历史记录失败，历史记录为空'
            }
        for _ in range(2):
            '''删除一轮对话'''
            first = self.history.messages.pop(0)
            print(f'删除上下文记忆: {first}')
        return {
            'status': 0,
            'message': '裁剪历史记录成功'
        }

    def talk(self, message, stream=False):
        message = message.replace("\n", " ")
        messages, history = p_generator.generate_ali_prompt(
            message=message, history=self.history.messages)
        if stream:
            total_reply = ''
            try:
                for response in Generation.call(
                    model='qwen-v1',
                    history=history,
                    prompt=messages,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    enable_search=self.kuake_search,
                    stream=True
                ):
                    if response.status_code == HTTPStatus.OK:
                        total_reply += response.output.text
                        yield {
                            'status': 0,
                            'message': response.output.text
                        }
                    else:
                        print('Error:Code: %d, status: %s, message: %s' % (
                            response.status_code, response.code, response.message))
                        yield {
                            'status': -1,
                            'message': 'Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message)
                        }
                        break
                self.history.messages.append(message)
                self.history.messages.append(total_reply)
            except:
                print('Code: %d, status: %s, message: %s' %
                      (response.status_code, response.code, response.message))
                yield {
                    'status': -1,
                    'message': 'Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message)
                }
        else:
            try:
                response = Generation.call(
                    model='qwen-v1',
                    history=history,
                    prompt=messages,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    enable_search=self.kuake_search,
                    stream=False
                )
                if response.status_code == HTTPStatus.OK:
                    self.history.messages.append(message)
                    self.history.messages.append(response.output.text)
                    yield {
                        'status': 0,
                        'message': response.output.text
                    }
                else:
                    print('Error:Code: %d, status: %s, message: %s' %
                          (response.status_code, response.code, response.message))
                    yield {
                        'status': -1,
                        'message': 'Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message)
                    }
            except:
                print('Error:Code: %d, status: %s, message: %s' %
                      (response.status_code, response.code, response.message))
                yield {
                    'status': -1,
                    'message': 'Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message)
                }

    def clear_history(self):
        self.history = ChatMessageHistory()


class spark_api():
    def __init__(self):
        self.history = ChatMessageHistory()

    def get_embedding(self):
        pass

    def setv(self, spark_api_key=None, spark_api_secret=None, spark_appid=None, temperature=0.95, top_k=4, max_tokens=2048, spark_api_version="V1.5"):
        if spark_appid == '':
            return {
                'status': -1,
                'message': '请输入spark appid'
            }
        if spark_api_key == '':
            return {
                'status': -1,
                'message': '请输入spar api key'
            }
        if spark_api_secret == '':
            return {
                'status': -1,
                'message': '请输入spark api secret'
            }
        self.spark_api_key = spark_api_key
        self.spark_api_secret = spark_api_secret
        self.spark_appid = spark_appid
        self.temperature = temperature
        self.top_k = top_k
        self.max_tokens = max_tokens
        if spark_api_version == "V1.5":
            self.domain = "general"
            self.gpt_url = "ws://spark-api.xf-yun.com/v1.1/chat"
        elif spark_api_version == "V2.0":
            self.domain = "generalv2"
            self.gpt_url = "ws://spark-api.xf-yun.com/v2.1/chat"
        else:
            return {
                "status": -1,
                "message": "API版本错误"
            }
        self.spark_api = Spark_Api(self.spark_appid, self.spark_api_secret, self.spark_api_key,
                                   self.temperature, self.top_k, self.max_tokens, self.gpt_url, self.domain)
        self.spark_api.create_ws()
        return {
            'status': 0,
            'message': '设置成功'
        }

    def get_ones(self, message):
        if type(message) == str:
            message = message.replace("\n", " ")
            message = p_generator.generate_spark_prompt(message=message)
        messages = message
        try:
            total_text = ''
            for response in self.spark_api._call(messages):
                if response['status'] == 0:
                    text = response['message']
                    total_text += text
                elif response['status'] == -1:
                    # print(response['message'])
                    return {
                        'status': -1,
                        'message': response['message']
                    }
            return {
                'status': 0,
                'message': total_text
            }
        except Exception as e:
            return {
                'status': -1,
                'message': str(e)
            }

    def get_ones_stream(self, message):
        if type(message) == str:
            message = message.replace("\n", " ")
            message = p_generator.generate_spark_prompt(message=message)
        messages = message
        try:
            for response in self.spark_api._call(messages):
                if response['status'] == 0:
                    text = response['message']
                    yield {
                        'status': 0,
                        'message': text
                    }
                elif response['status'] == -1:
                    # print(response['message'])
                    yield {
                        'status': -1,
                        'message': response['message']
                    }
                    break
        except Exception as e:
            yield {
                'status': -1,
                'message':  str(e)
            }

    def cut_memory(self):
        if len(self.history.messages) == 0:
            return {
                'status': -1,
                'message': '裁剪历史记录失败，历史记录为空'
            }
        for _ in range(2):
            '''删除一轮对话'''
            first = self.history.messages.pop(0)
            print(f'删除上下文记忆: {first}')
        return {
            'status': 0,
            'message': '裁剪历史记录成功'
        }

    def talk(self, message, stream=False):
        message = message.replace("\n", " ")
        messages = p_generator.generate_spark_prompt(
            message=message, history=self.history.messages)
        if stream:
            total_text = ''
            try:
                for response in self.spark_api._call(messages):
                    if response['status'] == 0:
                        text = response['message']
                        total_text += text
                        yield {
                            'status': 0,
                            'message': text
                        }
                    elif response['status'] == -1:
                        # print(response['message'])
                        yield {
                            'status': -1,
                            'message': response['message']
                        }
                        break
                self.history.messages.append(message)
                self.history.messages.append(total_text)
            except Exception as e:
                yield {
                    'status': -1,
                    'message': str(e)
                }
        else:
            try:
                total_text = ''
                for response in self.spark_api._call(messages):
                    if response['status'] == 0:
                        text = response['message']
                        total_text += text
                    elif response['status'] == -1:
                        # print(response['message'])
                        yield {
                            'status': -1,
                            'message': response['message']
                        }
                        break
                self.history.messages.append(message)
                self.history.messages.append(total_text)
                yield {
                    'status': 0,
                    'message': total_text
                }
            except Exception as e:
                yield {
                    'status': -1,
                    'message': str(e)
                }

    def clear_history(self):
        self.history = ChatMessageHistory()


class chatglm_api():
    def __init__(self):
        self.history = ChatMessageHistory()

    def get_embedding(self):
        pass

    def setv(self, api_key=None, temperature=0.95, top_p=0.7, chatglm_type='std'):
        if api_key == '':
            return {
                'status': -1,
                'message': '请输入chatglm api_key'
            }
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.chatglm_type = chatglm_type
        zhipuai.api_key = api_key
        return {
            'status': 0,
            'message': '设置成功'
        }

    def get_ones(self, message):
        if type(message) == str:
            message = message.replace("\n", " ")
            message = p_generator.generate_chatglm_prompt(message=message)
        messages = message
        try:
            response = zhipuai.model_api.invoke(
                model='chatglm_'+self.chatglm_type,
                prompt=messages,
                temperature=self.temperature,
                top_p=self.top_p)
            if response['success']:
                return {
                    'status': 0,
                    'message': response['data']['choices'][0]['content'][1:-1]
                }
            else:
                print(response)
                return {
                    'status': -1,
                    'message': 'code:%s, message:%s' % (response['code'], response['msg'])
                }
        except Exception as e:
            return {
                'status': -1,
                'message': str(e)
            }

    def get_ones_stream(self, message):
        if type(message) == str:
            message = message.replace("\n", " ")
            message = p_generator.generate_chatglm_prompt(message=message)
        messages = message
        try:
            response = zhipuai.model_api.sse_invoke(
                model='chatglm_'+self.chatglm_type,
                prompt=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                incremental=True
            )
            for event in response.events():
                if event.event == "add":
                    yield {
                        'status': 0,
                        'message': event.data
                    }
                elif event.event == "error" or event.event == "interrupted":
                    print(event.data)
                    yield {
                        'status': -1,
                        'message': event.data
                    }
                elif event.event == "finish":
                    yield {
                        'status': 0,
                        'message': event.data
                    }
                else:
                    print(event.data)
                    yield {
                        'status': -1,
                        'message': event.data
                    }
        except Exception as e:
            print(response)
            yield {
                'status': -1,
                'message': str(e)
            }

    def cut_memory(self):
        if len(self.history.messages) == 0:
            return {
                'status': -1,
                'message': '裁剪历史记录失败，历史记录为空'
            }
        for _ in range(2):
            '''删除一轮对话'''
            first = self.history.messages.pop(0)
            print(f'删除上下文记忆: {first}')
        return {
            'status': 0,
            'message': '裁剪历史记录成功'
        }

    def talk(self, message, stream=False):
        message = message.replace("\n", " ")
        messages = p_generator.generate_chatglm_prompt(
            message=message, history=self.history.messages)
        if stream:
            total_reply = ''
            try:
                for response in self.get_ones_stream(messages):
                    if response['status'] == 0:
                        text = response['message']
                        total_reply += text
                        yield {
                            'status': 0,
                            'message': text
                        }
                    elif response['status'] == -1:
                        print(response['message'])
                        yield {
                            'status': -1,
                            'message': response['message']
                        }
                        break
                self.history.messages.append(message)
                self.history.messages.append(total_reply)
            except:
                yield {
                    'status': -1,
                    'message': response['message']
                }
        else:
            try:
                response = self.get_ones(messages)
                if response['status'] == 0:
                    reply = response['message']
                    self.history.messages.append(message)
                    self.history.messages.append(reply)
                    yield {
                        'status': 0,
                        'message': reply
                    }
                elif response['status'] == -1:
                    print(response['message'])
                    yield {
                        'status': -1,
                        'message': response['message']
                    }
            except:
                yield {
                    'status': -1,
                    'message': response['message']
                }

    def clear_history(self):
        self.history = ChatMessageHistory()


class ernie_api():
    def __init__(self) -> None:
        self.history = ChatMessageHistory()

    def get_embedding(self):
        pass

    def setv(self, ernie_api_key=None, ernie_secret_key=None, ernie_temperature=0.95, ernie_top_p=0.8, ernie_penalty_score=1, ernie_type='ernie bot'):
        if ernie_api_key == '':
            return {
                'status': -1,
                'message': '请输入ernie api key'
            }
        if ernie_secret_key == '':
            return {
                'status': -1,
                'message': '请输入ernie secret key'
            }
        self.ernie_type = ernie_type
        if ernie_type == 'ernie bot':
            self.url = f'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions'
        elif ernie_type == 'ernie bot turbo':
            self.url = f'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant'
        else:
            return {
                'status': -1,
                'message': '请输入正确的ernie类型'
            }

        self.headers = {
            'Content-Type': 'application/json',
        }
        self.temperature = ernie_temperature
        self.top_p = ernie_top_p
        self.penalty_score = ernie_penalty_score

        self.access_token = get_ernie_access_token(
            ernie_api_key, ernie_secret_key)
        if self.access_token == None:
            return {
                'status': -1,
                'message': '获取ernie access_token失败'
            }

        self.query = {
            'access_token': self.access_token
        }

        return {
            'status': 0,
            'message': '设置成功'
        }

    def get_ones(self, message):
        if type(message) == str:
            message = message.replace("\n", " ")
            message = p_generator.generate_ernie_prompt(message=message)
        if self.ernie_type == 'ernie bot':
            body = {
                'messages': message,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'penalty_score': self.penalty_score,
                'stream': False
            }
        elif self.ernie_type == 'ernie bot turbo':
            body = {
                'messages': message,
                'stream': False
            }
        try:
            response = requests.request(
                "POST", self.url, headers=self.headers, params=self.query, data=json.dumps(body))
            response = json.loads(response.text)
            if response.get('result', None) is not None:
                return {
                    'status': 0,
                    'message': response['result']
                }
            else:
                print('code:%s message:%s' %
                      (response['error_code'], response['error_msg']))
                return {
                    'status': -1,
                    'message': 'code:%s message:%s' % (response['error_code'], response['error_msg'])
                }
        except Exception as e:
            return {
                'status': -1,
                'message': str(e)
            }

    def get_ones_stream(self, message):
        if type(message) == str:
            message = message.replace("\n", " ")
            message = p_generator.generate_ernie_prompt(message=message)
        if self.ernie_type == 'ernie bot':
            body = {
                'messages': message,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'penalty_score': self.penalty_score,
                'stream': True
            }
        elif self.ernie_type == 'ernie bot turbo':
            body = {
                'messages': message,
                'stream': True
            }

        try:
            response = requests.request(
                "POST", self.url, headers=self.headers, params=self.query, data=json.dumps(body))
            for section in response.iter_lines(decode_unicode=True):
                section = json.loads(section)
                if section.get('result', None) is not None:
                    yield {
                        'status': 0,
                        'message': section['result']
                    }
                else:
                    print('code:%s message:%s' %
                          (section['error_code'], section['error_msg']))
                    yield {
                        'status': -1,
                        'message': 'code:%s message:%s' % (section['error_code'], section['error_msg'])
                    }
        except Exception as e:
            yield {
                'status': -1,
                'message': str(e)
            }

    def cut_memory(self):
        if len(self.history.messages) == 0:
            return {
                'status': -1,
                'message': '裁剪历史记录失败，历史记录为空'
            }
        for _ in range(2):
            '''删除一轮对话'''
            first = self.history.messages.pop(0)
            print(f'删除上下文记忆: {first}')
        return {
            'status': 0,
            'message': '裁剪历史记录成功'
        }

    def talk(self, message, stream=False):
        message = message.replace("\n", " ")
        messages = p_generator.generate_ernie_prompt(
            message=message, history=self.history.messages)
        if stream:
            total_reply = ''
            try:
                for response in self.get_ones_stream(messages):
                    if response['status'] == 0:
                        text = response['message']
                        total_reply += text
                        yield {
                            'status': 0,
                            'message': text
                        }
                    elif response['status'] == -1:
                        print(response['message'])
                        yield {
                            'status': -1,
                            'message': response['message']
                        }
                        break
                self.history.messages.append(message)
                self.history.messages.append(total_reply)
            except Exception as e:
                yield {
                    'status': -1,
                    'message': str(e)
                }
        else:
            try:
                response = self.get_ones(messages)
                if response['status'] == 0:
                    reply = response['message']
                    self.history.messages.append(message)
                    self.history.messages.append(reply)
                    yield {
                        'status': 0,
                        'message': reply
                    }
                elif response['status'] == -1:
                    # print(response['message'])
                    yield {
                        'status': -1,
                        'message': response['message']
                    }
            except Exception as e:
                yield {
                    'status': -1,
                    'message': str(e)
                }

    def clear_history(self):
        self.history = ChatMessageHistory()


class openai_api():
    def __init__(self):
        self.history = ChatMessageHistory()

    def get_embedding(self, openai_api_key, port='', api_base='', api_model='text-embedding-ada-002', type='openai', endpoint='', engine=''):
        if type == 'openai':
            openai.api_type = "open_ai"
            if api_base != '':
                openai.api_base = api_base
            else:
                openai.api_base = 'https://api.openai.com/v1'
            if port != '':
                os.environ['http_proxy'] = 'http://127.0.0.1:' + port
                os.environ["https_proxy"] = "http://127.0.0.1:" + port
            self.embedding = OpenAIEmbeddings(openai_api_key=openai_api_key, model=api_model)
        elif type == 'azure':
            os.environ['OPENAI_API_BASE'] = endpoint
            os.environ['OPENAI_API_KEY'] = openai_api_key
            os.environ['OPENAI_API_TYPE'] = 'azure'
            # 暂未支持
            self.embedding = OpenAIEmbeddings(
                openai_api_key=openai_api_key, deployment=engine, model=api_model)

    def setv(self, openai_api_key='', api_base='', temperature=0.95, max_tokens=None, top_p=0.7, openai_prompt='', port=10809, model="gpt-3.5-turbo", type='openai', endpoint='', engine=""):
        if openai_api_key == '':
            return {
                'status': -1,
                'message': '请输入openai api key'
            }
        if type == 'openai':
            openai.api_type = "open_ai"
            if api_base != '':
                openai.api_base = api_base
            else:
                openai.api_base = 'https://api.openai.com/v1'
            # if port == '':
            #     return {
            #         'status': -1,
            #         'message': '请输入openai代理端口'
            #     }
            if port != '':
                os.environ['http_proxy'] = 'http://127.0.0.1:'+port
                os.environ["https_proxy"] = "http://127.0.0.1:"+port
        elif type == 'azure':
            if endpoint == '':
                return {
                    'status': -1,
                    'message': '请输入openai endpoint'
                }
            if engine == '':
                return {
                    'status': -1,
                    'message': '请输入openai engine'
                }
        self.charactor_prompt = SystemMessage(content=openai_prompt)
        self.max_token = max_tokens
        if max_tokens == 0 or max_tokens == None:
            max_tokens = None
            self.max_token = 4096
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
                max_tokens=max_tokens)
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
                max_tokens=max_tokens)
            # 暂未支持
            self.embedding = OpenAIEmbeddings(
                openai_api_key=openai_api_key, deployment=engine)
        return {
            'status': 0,
            'message': '设置成功'
        }

    def get_ones(self, text):
        if type(text) == str:
            text = text.replace("\n", " ")
            text = p_generator.generate_openai_prompt(
                message=text, system_message=self.charactor_prompt.content)
        messages = text
        try:
            if self.llm.get_num_tokens_from_messages(messages) >= self.max_token-500:
                return '字数超限'
        except Exception as e:
            pass
        try:
            response = self.llm_nonstream(messages)
            # print(response.content)
            return {
                'status': 0,
                'message': response.content
            }
        except Exception as e:
            return {
                'status': -1,
                'message': str(e)
            }

    def get_ones_stream(self, text):
        if type(text) == str:
            text = text.replace("\n", " ")
            text = p_generator.generate_openai_prompt(
                message=text, system_message=self.charactor_prompt.content)
        messages = text
        try:
            if self.llm.get_num_tokens_from_messages(messages) >= self.max_token-500:
                return '字数超限'
        except Exception as e:
            pass
        try:
            for response in self.llm(messages):
                # print(response[1])
                if len(response) == 0 or type(response[1]) == dict:
                    break
                yield {
                    'status': 0,
                    'message': response[1]
                }
        except Exception as e:
            yield {
                'status': -1,
                'message': str(e)
            }

    def cut_memory(self):
        if len(self.history.messages) == 0:
            return {
                'status': -1,
                'message': '裁剪历史记录失败，历史记录为空'
            }
        for _ in range(2):
            '''删除一轮对话'''
            first = self.history.messages.pop(0)
            print(f'删除上下文记忆: {first}')
        return {
            'status': 0,
            'message': '裁剪历史记录成功'
        }

    def talk(self, message, stream=False):
        message = message.replace("\n", " ")
        messages = p_generator.generate_openai_prompt(
            message=message, system_message=self.charactor_prompt.content, history=self.history.messages)
        try:
            if self.llm.get_num_tokens_from_messages(messages) >= self.max_token-500:
                self.cut_memory()
        except Exception as e:
            pass
        if stream:
            total_reply = ''
            try:
                for response in self.llm(messages):
                    if len(response) == 0 or type(response[1]) == dict:
                        break
                    total_reply += response[1]
                    yield {
                        'status': 0,
                        'message': response[1]
                    }
                self.history.messages.append(HumanMessage(content=message))
                self.history.messages.append(AIMessage(content=total_reply))
            except Exception as e:
                yield {
                    'status': -1,
                    'message': str(e)
                }
        else:
            try:
                response = self.llm_nonstream(messages).content
                self.history.messages.append(HumanMessage(content=message))
                self.history.messages.append(AIMessage(content=response))
                yield {
                    'status': 0,
                    'message': response
                }
            except Exception as e:
                yield {
                    'status': -1,
                    'message': str(e)
                }

    def clear_history(self):
        self.history = ChatMessageHistory()
