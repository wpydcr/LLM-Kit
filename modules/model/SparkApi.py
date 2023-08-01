import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from queue import Queue
import websocket
import gradio as gr


class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, gpt_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(gpt_url).netloc
        self.path = urlparse(gpt_url).path
        self.gpt_url = gpt_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(
            signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(
            authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.gpt_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url


class Spark_Api(object):
    def __init__(self, appid,
                 api_secret,
                 api_key,
                 temperature=0.5,
                 top_k=4,
                 max_tokens=2048,
                 gpt_url="ws://spark-api.xf-yun.com/v1.1/chat"):
        self.appid = appid
        self.api_secret = api_secret
        self.api_key = api_key
        self.temperature = temperature
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.gpt_url = gpt_url
        self.queue = Queue()

    def create_ws(self):
        self.wsParam = Ws_Param(self.appid, self.api_key,
                                self.api_secret, self.gpt_url)
        websocket.enableTrace(False)
        self.wsUrl = self.wsParam.create_url()
        self.ws = websocket.WebSocketApp(
            self.wsUrl, on_message=self.on_message, on_error=self.on_error, on_close=self.on_close, on_open=self.on_open)

    def _call(self, messages):
        self.messages = messages
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        while True:
            text = self.queue.get()
            if text['status'] == 0 and text['message'] is None:
                break
            yield text
            if text['status'] == -1:
                break

    # 收到websocket错误的处理
    def on_error(self, ws, error):
        self.queue.put({
            'status': -1,
            'message': str(error)
        })

    # 收到websocket关闭的处理
    def on_close(self, ws, *args):
        self.queue.put({
            'status': -1,
            'message': 'websocket closed'
        })

    # 收到websocket连接建立的处理
    def on_open(self, ws, *args):
        thread.start_new_thread(self.run, (ws, ))

    def run(self, ws, *args):
        data = json.dumps(self.gen_params())
        self.ws.send(data)

    # 收到websocket消息的处理
    def on_message(self, ws, message):

        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            # print(f'请求错误: {code}, {data}')
            ws.close()
            self.queue.put({
                'status': -1,
                'message': message
            })
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            self.queue.put({
                'status': 0,
                'message': content
            })
            if status == 2:
                self.queue.put({
                    'status': 0,
                    'message': None
                })
                ws.close()

    def gen_params(self):
        """
        通过appid和用户的提问来生成请参数
        """

        data = {
            "header": {
                "app_id": self.appid,
                "uid": "1234"
            },
            "parameter": {
                "chat": {
                    "domain": "general",
                    "auditing": "default",
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_k": self.top_k,
                }
            },
            "payload": {
                "message": {
                    "text": self.messages
                }
            }
        }
        return data