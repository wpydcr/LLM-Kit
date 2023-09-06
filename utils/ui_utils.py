# This file contains classes that response to the train module on the webui, to keep tracing states, e.g., the loaded
# model, instantiation the class and bound the response function to the webui event, click, change for instance.
# 这个文件内包含了响应webui界面上“train”模块各种行为的python类。若需要保存对应类的状态，列如导入的模型，需要实例化对应的类，随后将对应的响应方法
# 绑定到webui 组件的事件上，例如按钮点击，选项变更。


import sys
import gradio as gr
from utils.dl_data import *
from collections import OrderedDict
from modules.model.use_api import *
from modules.apply.role_play import role_play
import threading
import queue
import time
from modules.model.llm_auto import AutoLM
import os
import numpy as np
import yaml
import random
from torch.cuda import device_count, is_available
from modules.agent.internet_search import internet_search
import re
from utils.embedding_base_model import embedding_base_model
from glob import glob
from modules.model.LLM import train_luancher as tl
from modules.agent.tts_online import get_voice
import datetime
import pandas as pd
import librosa
from modules.agent.chatdb.mysql import MySQLDB
from utils.utils import parse_input_string
from utils.svc_utils import SVC
from utils.vits_utils import VITS
from modules.agent.chatdb.chatdb import generate_chat_responses

real_path = os.path.split(os.path.realpath(__file__))[0]


def parse_text(text):
    """Parses text and converts markdown code blocks to HTML code blocks.

    Args:
        text (str): The input text.

    Returns:
        str: The parsed text.

    """
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


class StoppableThread(threading.Thread):
    """Thread class customized to enable stop and restart for data processing function
        in webui data module.
    """

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()
        self.queue = queue.Queue()
        self.openai = None
        self.timesleep = 20
        self.answer_list = []
        self.flag = False

    def setv(self, openai_api, timesleep, answer_list, type_=None):
        self.openai = openai_api
        self.timesleep = timesleep
        self.answer_list = answer_list
        self.type = type_

    def stop(self):
        if not self.stopped():
            self.flag = False

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self.flag = True
        if self.type is None:
            while not self.stopped() and len(self.answer_list) > 0:
                ans = self.openai.get_ones(self.answer_list[0])
                if ans['status'] == 0:
                    ans = ans['message']
                else:
                    raise gr.Error(ans['message'])
                self.queue.put([self.answer_list[0], ans, self.answer_list])
                self.answer_list = self.answer_list[1:]
                time.sleep(self.timesleep)
            self.queue.put(None)
        elif self.type == 'embed':
            while not self.stopped() and len(self.answer_list) > 0:
                inputs = self.answer_list[0][0]+'。'+self.answer_list[0][1]
                ans = self.openai.get_ones(inputs)
                if ans['status'] == 0:
                    ans = ans['message']
                else:
                    raise gr.Error(ans['message'])
                self.queue.put([inputs.split('。'), ans, self.answer_list])
                self.answer_list = self.answer_list[1:]
                time.sleep(self.timesleep)
            self.flag = False


class data_process():
    """Class for data processing operations.

    Attributes:
        json_dict (OrderedDict): The dictionary to store JSON data.
        embedding_json_dict (OrderedDict): The dictionary to store embedding JSON data.
        org_json_dict (OrderedDict): The dictionary to store original JSON data.
        answer_list (list): The list to store answers.
        sentence_pair (list): The list to store sentence pairs.
        thread (StoppableThread): The thread for asynchronous processing.
        upload_dict (OrderedDict): The dictionary to store uploaded data.
        openai (openai_api): The OpenAI API object.
        mysql (MySQLDB): The MySQLDB object.

    """

    def __init__(self):
        self.json_dict = OrderedDict()
        self.embedding_json_dict = OrderedDict()
        self.org_json_dict = OrderedDict()
        self.answer_list = []
        self.sentence_pair = []
        self.thread = StoppableThread()
        self.upload_dict = OrderedDict()
        self.openai = openai_api()
        self.mysql = MySQLDB()

    def save_data(self, a, b):
        c = ''
        d = ''
        if a != '' and b != '':
            if len(self.answer_list) > 1:
                self.answer_list = self.answer_list[1:]
                c = self.answer_list[0]
            self.json_dict[str(a)] = str(b)
        elif a != '':
            c = a
        elif b != '':
            d = b
        return self.json_dict, c, d

    def skip_qa(self, a):
        if len(self.answer_list) > 1:
            self.answer_list = self.answer_list[1:]
            return self.answer_list[0], a
        return '', ''

    def save_embed_data(self, sentence1, sentence2, label):
        c = ''
        d = ''
        if sentence1 != '' and sentence2 != '':
            self.embedding_json_dict[str(len(self.embedding_json_dict)+1)] = {
                'sentence1': sentence1,
                'sentence2': sentence2,
                'label': 1 if label == '相关' else 0
            }
            if len(self.sentence_pair) > 0:
                self.sentence_pair = self.sentence_pair[1:]
                c = self.sentence_pair[0][0]
                d = self.sentence_pair[0][1]
        elif sentence1 != '':
            c = sentence1
        elif sentence2 != '':
            d = sentence2
        return self.embedding_json_dict, c, d

    def back_embed_json(self):
        if len(self.embedding_json_dict) > 0:
            self.embedding_json_dict.popitem(last=True)
        return self.embedding_json_dict

    def empty_embed_json(self):
        self.embedding_json_dict = OrderedDict()
        return self.embedding_json_dict

    def empty_embed_upload(self):
        self.sentence_pair = []
        return '', ''

    def empty_embed_exchange_upload(self):
        self.embedding_json_dict = OrderedDict()
        return self.embedding_json_dict

    def reset_state(self):
        self.json_dict = OrderedDict()
        return self.json_dict

    def reset_upload(self):
        self.answer_list = []
        return ''

    def reset_upload_out(self):
        self.upload_dict = OrderedDict()
        return self.upload_dict

    def reset_state_doc(self):
        self.json_dict = OrderedDict()
        self.org_json_dict = OrderedDict()
        return self.json_dict, self.org_json_dict

    def back_state(self):
        if len(self.json_dict) > 0:
            self.json_dict.popitem(last=True)
        return self.json_dict

    def upload_data(self, temp_file):
        fname = temp_file.name
        with open(fname, 'r', encoding='utf-8') as f:
            self.answer_list = [i.strip() for i in f.readlines()]
        return self.answer_list[0]

    def upload_data_out(self, temp_file, a, b, models2, models3):
        if models3 == '新建':
            self.upload_dict = OrderedDict()
        fname = temp_file.name
        if models2 == 'json':
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    lsb = json.load(f)
                    if a in lsb[0].keys() and b in lsb[0].keys():
                        for example in lsb:
                            self.upload_dict[example[a]] = example[b]
            except:
                raise gr.Error('data error')
        else:
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        example = json.loads(line)
                        if a in example.keys() and b in example.keys():
                            self.upload_dict[example[a]] = example[b]
            except:
                raise gr.Error('data error')
        oute = list(self.upload_dict.items())
        return [['数据集大小', str(len(oute))+'条']]+oute[-6:]

    def upload_embed_data(self, embed_upload):
        self.sentence_pair = []
        fname = embed_upload.name
        with open(fname, 'r', encoding='utf-8') as f:
            content = f.read()
            content = content.replace('\n', '')
            content = re.sub('\s+', '', content)
            sentences = content.split('。')
        if len(sentences) % 2 != 0:
            sentences = sentences[:-1]
        for i in range(0, len(sentences), 2):
            self.sentence_pair.append([sentences[i], sentences[i+1]])
        return self.sentence_pair[0][0], self.sentence_pair[0][1]

    def upload_embed_exchange_data(self, embed_exchange_upload, type_):
        if type_ == '新建':
            self.embedding_json_dict = OrderedDict()
        fname = embed_exchange_upload.name
        with open(fname, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.split()) == 3 and line.split()[2].isdigit():
                    self.embedding_json_dict[str(len(self.embedding_json_dict))] = {
                        "sentence1": line.split()[0],
                        "sentence2": line.split()[1],
                        "label": 0 if int(line.split()[2]) < 2.5 else 1
                    }
                else:
                    raise gr.Error(
                        "Unsupported format of file.请检查上传的文本格式是否符合要求。")
        outs = list(self.embedding_json_dict.items())
        return [["数据集大小", str(len(outs))+'条']]+outs[-6:]

    def ones_openai(self, question, openai_api, temperature, max_tokens, top_p, openai_prompt, port):
        if len(openai_api) < 40 or question == '':
            return None
        response = self.openai.setv(openai_api, temperature, max_tokens,
                         top_p, openai_prompt, port)
        if response['status'] == -1:
            raise gr.Error(response['message'])
        ans = self.openai.get_ones(question)
        if ans['status'] == 0:
            ans = ans['message']
        else:
            raise gr.Error(ans['message'])
        return ans

    def ones_openai_doc(self, question, openai_api, temperature, max_tokens, top_p, openai_prompt, port):
        response = self.openai.setv(openai_api, temperature, max_tokens,
                         top_p, openai_prompt, port)
        if response['status'] == -1:
            raise gr.Error(response['message'])
        ans = self.openai.get_ones(question)
        if ans['status'] == 0:
            ans = ans['message']
        else:
            raise gr.Error(ans['message'])
        self.org_json_dict[question] = ans
        for i in ans.split('。'):
            j = i.split('答：')
            try:
                self.json_dict[j[0].strip()[2:]] = j[1].strip() + '。'
            except:
                pass
        return self.json_dict, self.org_json_dict

    def start_openai_doc(self, openai_api, temperature, max_tokens, top_p, openai_prompt, port, timesleep):
        response = self.openai.setv(openai_api, temperature, max_tokens,
                         top_p, openai_prompt, port)
        if response['status'] == -1:
            raise gr.Error(response['message'])
        self.thread.setv(self.openai, timesleep, self.answer_list)
        # 启动线程
        self.thread.start()
        while True:
            item = self.thread.queue.get()
            if item is None:
                break
            self.answer_list = item[2]
            self.org_json_dict[str(item[0])] = str(item[1])
            for i in item[1].split('。'):
                j = i.split('答：')
                try:
                    self.json_dict[j[0].strip()[2:]] = j[1].strip() + '。'
                except:
                    pass
            yield gr.update(value=self.json_dict), gr.update(value=self.org_json_dict)
        self.thread = StoppableThread()
        return self.json_dict

    def start_openai(self, openai_api, temperature, max_tokens, top_p, openai_prompt, port, timesleep):
        response = self.openai.setv(openai_api, temperature, max_tokens,
                         top_p, openai_prompt, port)
        if response['status'] == -1:
            raise gr.Error(response['message'])
        self.thread.setv(self.openai, timesleep, self.answer_list)
        # 启动线程
        self.thread.start()
        while True:
            item = self.thread.queue.get()
            if item is None:
                break
            self.answer_list = item[2]
            self.json_dict[str(item[0])] = str(item[1])
            yield gr.update(value=self.json_dict)
        self.thread = StoppableThread()
        return self.json_dict

    def start_openai_embed(self, openai_api, temperature, max_tokens, top_p, openai_prompt, port, timesleep):
        response = self.openai.setv(openai_api, temperature, max_tokens,
                         top_p, openai_prompt, port)
        if response['status'] == -1:
            raise gr.Error(response['message'])
        self.thread.setv(self.openai, timesleep,
                         self.sentence_pair, type_='embed')
        # 启动线程
        self.thread.start()
        while True:
            if self.thread.flag:
                item = self.thread.queue.get()
            if self.thread.flag == False:
                break
            self.sentence_pair = item[2]
            self.embedding_json_dict[str(len(self.embedding_json_dict))] = {
                'sentence1': item[0][0],
                'sentence2': item[0][1],
                'label': 1 if '1' in item[1] else 0
            }
            yield self.embedding_json_dict, self.sentence_pair[0][0], self.sentence_pair[0][1]

        self.thread = StoppableThread()
        # return self.embedding_json_dict, self.answer_list[0][0], self.answer_list[0][1]

    def stop_openai_embed(self):
        self.thread.stop()
        # self.thread = StoppableThread()
        return ''

    def stop_openai(self):
        self.thread.stop()
        self.thread = StoppableThread()
        return self.json_dict, self.answer_list[0] if len(self.answer_list) > 0 else ''

    def stop_openai_doc(self):
        self.thread.stop()
        self.thread = StoppableThread()
        return self.json_dict, self.org_json_dict, self.answer_list[0] if len(self.answer_list) > 0 else ''

    # ------------------ download data -------------------
    def split_json(self, json_l, ratio=0.2):
        keys = list(json_l.keys())
        random.shuffle(keys)

        split_point = int(len(keys) * ratio)
        json1_keys = keys[:split_point]
        json2_keys = keys[split_point:]

        json_test = {key: json_l[key] for key in json1_keys}
        json_train = {key: json_l[key] for key in json2_keys}
        return [json_test, json_train]

    def dl_jsonl1(self, dpath, split_data):
        if self.json_dict == {}:
            return gr.update(value="No data")
        if split_data:
            download_jsonl_data(self.split_json(self.json_dict), dpath)
        else:
            download_jsonl_data([self.json_dict], dpath)
        return gr.update(value="Saved to data/modeldata/LLM")

    def dl_embed(self, embed_download_path, train_valid=False):
        if self.embedding_json_dict == {}:
            return gr.update(value="No data")
        if train_valid:
            download_jsonl_data(self.split_json(
                self.embedding_json_dict), embed_download_path, 'embed')
        else:
            download_jsonl_data([self.embedding_json_dict],
                                embed_download_path, 'embed')
        return gr.update(value="Saved to data/modeldata/Embedding")

    def dl_jsonl2(self, dpath, split_data):
        if self.upload_dict == {}:
            return gr.update(value="No data")
        if split_data:
            download_jsonl_data(self.split_json(self.upload_dict), dpath)
        else:
            download_jsonl_data([self.upload_dict], dpath)
        return gr.update(value="Saved to data/modeldata/LLM")

    def connect_mysql(self, host, user, password, port):
        try:
            self.mysql.connect(host=host, user=user,
                               password=password, port=port)
            databases = self.mysql.get_databases()
            return gr.update(choices=databases+['新建数据库'])
        except:
            raise gr.Error('连接失败，请检查数据库配置是否正确')

    def change_database(self, database):
        if database == '新建数据库':
            return gr.update(visible=True)
        return gr.update(visible=False)

    def mysql_upload(self, host, user, password, port, database, new_database, files):
        # 支持excel、csv、txt
        error = []
        success = []
        if database != '新建数据库':
            new_database = database
        else:
            self.mysql.connect(host=host, user=user,
                               password=password, port=port)
            if not self.mysql.create_database(new_database):
                raise gr.Error('数据库创建失败')
        for file in files:
            name = file.name
            if name.split('.')[-1] == 'xlsx' or name.split('.')[-1] == 'xls':
                df = pd.read_excel(name, header=0, engine='openpyxl')
            elif name.split('.')[-1] == 'csv':
                df = pd.read_csv(name, header=0, delimiter=',')
            elif name.split('.')[-1] == 'txt':
                df = pd.read_table(name, header=0, delimiter=' ')
            else:
                raise gr.Error('文件格式不正确，请检查文件格式')
            flag = self.mysql.load_data(new_database, os.path.split(
                name)[1].split('.')[0], df.columns, df.values.tolist())
            if not flag:
                error.append(os.path.split(name)[1].split('.')[0])
            else:
                success.append(os.path.split(name)[1].split('.')[0])
        msg = ''
        for i in success:
            msg += f'{i} 文件上传成功\n'
        for i in error:
            msg += f'{i} 文件上传失败\n'
        return gr.update(value=msg)


class video_apply():
    """Class responses to the apply module on the webui page. It implemented the setup of chat language models, including
    local models and openai api, conversation inference and text to voice conversion.

    Attributes:
        chat (play_base_api): The play_base_api object.
        history (list): The list to store chat history.
        svc (SVC): The SVC object for voice processing, used when voice style is selected.
        vits (VITS): The VITS object for text to voice generation, used when local voice generation is selected.

    """

    def __init__(self):
        self.chat = play_base_api()
        self.history = []
        self.svc = None
        self.vits = None

    def set_v(self, params):
        """Sets the parameters for play_base_api.

                Args:
                    openai_api_key (str): The OpenAI API key.
                    play (str): The cosplay character name, e.g., 坂田银时.
                    user_name (str): The appellation user chose to be called during conversation.
                    memory (str): The memory setting file.
                    port (int): The port number.
                    back (str): The background setting file.
                    net (bool): If enable online searching in conversation.
                    search (str): The search engine chose, e.g., google/bing....
                    search_key (str): The search key applied from search engine.
                    result_len (int): The result length.
                    emb (bool): The embedding model chosen.
                    emoticon (bool): The emoticon value.
                    time_c (bool): Enable time perception for chat language model.

                Returns:
                    str: Empty string.

                Raises:
                    gr.Error: If `play` is None or both `net` and `search` are None.

        """
        self.chat.set_v(params)
        return True

    def set_llm(self, params):
        """Sets the parameters for play_base_api large language Model.

        Args:
            openai_api_key (str): The OpenAI API key.
            temperature: The temperature for generation, influences the randomness of generation.
            max_tokens: The maximum number of tokens.
            top_p: The number of results to be matched in vector store searching..
            port (int): The port number.
            models0 (str): The model name.
            lora0: The lora model name, null if not specificed.
            endpoint: The endpoint value.
            engine: The engine value.

        Returns:
            str: Empty string.

        Raises:
            gr.Error: If `models0` is not selected.

        """
        self.chat.set_llm(params)
        return '', gr.update(visible=True)

    def predict(self, input, chatbot, gen_type, lang, voice_style, show_type):
        """Generates inference using the large language Model by given input as prompt.

                Args:
            input (str): The input for prediction.
            chatbot (list): The chatbot conversation list.
            gen_type (str): The generation type, local or online.
            lang (str): The output language, e.g., 中文、英语、日语.
            voice_style (str): The voice style, used for SVC.

        Yields:
            Tuple[str, list, str, str]: Empty string, updated chatbot conversation list, loud values, and audio path.

        Raises:
            gr.Error: If the LL Model or character prompt is not set.

        """
        if self.chat.llm.play is None:
            raise gr.Error('请选择设定')
        response, emo = self.chat.llm.talk(input)
        chatbot.append((input, response+('' if emo is None else emo)))
        if show_type == '文本':
            return '', chatbot, 's', None
        else:
            loud, audio = self.generate_video(
                response, gen_type, lang, voice_style)
            return '', chatbot, loud, audio

    def generate_video(self, response, gen_type, lang, voice_style):
        """Used to use to generate video that match the response audio, abandoned and replaced by live2d character.
            Now only generates the voice base on the inference content from the model, and process the audio to
            the chosen speaker voice style if svc voice style is selected.

                Args:
                    response (str): The response form the llm.
                    gen_type (str): The generation type, e.g., 在线、本地、keqing...
                    lang (str): The output language.
                    voice_style (str): The voice style.

                Returns:
                    Tuple[str, str]: Loud values and audio path.

                """

        response = parse_input_string(response)

        filename = os.path.join(
            real_path, "..", "data", "apply", "audio", "tts")

        if gen_type == "本地" or gen_type == "在线":
            out_f = get_voice(response, 5, filename=filename,
                              gen_type=gen_type, lang=lang)
        else:
            if self.vits == None:
                self.vits = VITS()
            self.vits.load(gen_type, 0)
            out_f = self.vits.to_audio(response, filename)

        if self.svc == None:
            self.svc = SVC()
        self.svc.load(voice_style)
        # doesnt process if the voice_style is default.
        self.svc.voice_process(voice_style, filename)

        if out_f != None:
            x, sr = librosa.load(os.path.join(
                real_path, "..", "data", "apply", "audio", "tts.mp3" if gen_type == '在线' else 'tts.wav'), sr=8000)

            x = x - min(x)
            x = x / max(x)
            x = np.log(x) + 1
            x = x / max(x) * 1.2

            loud = []
            s_time = time.time()
            try:
                for _ in range(int(len(x) / 800)):
                    it = x[int((time.time() - s_time) * 8000)+1]
                    if it < 0:
                        it = 0

                    loud.append(float(it))
                    time.sleep(0.1)
            except:
                pass
            loud.append(0)

            return str(loud), os.path.join(real_path, "..", "data", "apply", "audio", "tts.mp3" if gen_type == '在线' else 'tts.wav')
        else:
            return None, None

    def reset_state(self):
        self.chat.llm.clear_history()
        return '', []

    def clear(self):
        if self.chat.llm.model_use is None:
            return [], '', gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(visible=False)
        if self.chat.llm.model_use == 'openai' or self.chat.llm.model_use == 'azure openai':
            self.chat.llm.model_use = None
            self.chat.llm.clear_history()
        else:
            self.chat.llm.model_use = None
            self.chat.llm.llm.clear()
            self.chat.llm.clear_history()
        return [], '', gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(visible=False)

    def clear_config(self):
        return gr.update(value=None), gr.update(value=False), gr.update(value=0), gr.update(value=''), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=False), gr.update(value='')


class play_base_api():
    """Class implemented the detailed local llm driven chatbot setup and inference."""

    def __init__(self):
        self.llm = role_play()

    def set_v(self, params):
        self.llm.setv(params)
        return True

    def set_llm(self, params):
        self.llm.set_llm(params)
        return '', []

    def predict(self, input, chatbot):
        responses = self.llm.talk(input)
        chatbot.append((parse_text(input), ""))
        for response in responses:
            chatbot[-1] = [parse_text(input), parse_text(response)]
            yield chatbot, ''

    def reset_state(self):
        self.llm.clear_history()
        return [], ''

    def clear(self):
        if self.llm.model_use == 'openai' or self.llm.model_use == 'azure openai':
            self.llm.model_use = ''
            self.llm.clear_history()
        else:
            if self.llm.model_use != '':
                self.llm.llm.clear()
                self.llm.clear_history()
        return [], ''


class chat_base_api():
    """Class implemented the detailed API driven chatbot model setup and inference."""

    def __init__(self, api_type):
        self.api_type = api_type
        if api_type == 'openai' or api_type == 'azure openai':
            self.llm = openai_api()
        elif api_type == 'ernie bot' or api_type == 'ernie bot turbo':
            self.llm = ernie_api()
        elif api_type == 'chatglm api':
            self.llm = chatglm_api()
        elif api_type == 'spark api':
            self.llm = spark_api()
        elif api_type == 'ali api':
            self.llm = ali_api()
        else:
            pass

    def set_v(self, params):
        """
                       Sets the necessary variables for the api chatbot model.

                       Args:
                           self: The play_base_api object itself.
                           params:
                               play (str): The cosplay character setting name.
                               user_name (str): The appellation user chose to be called during conversation.
                               memory (dict): The memory dictionary.
                               net (str): The network string.
                               search (str): The search string.
                               search_key (str): The search key string.
                               result_len (int): The length of search results.
                               back (int): The number of backward steps in the conversation.
                               emb (str): The embedding model.
                               emoticon (str): The emoticon string.
                               time_c (bool): The time proception.
                               openai_api_key (str): The OpenAI API key.
                               port (int): The port.

               """
        if self.api_type == 'openai':
            response = self.llm.setv(openai_api_key=params['api_key'], openai_prompt=params.get(
                'prompt', ''), port=params['port'])
        elif self.api_type == 'azure openai':
            response = self.llm.setv(openai_api_key=params['api_key'], openai_prompt=params.get(
                'prompt', ''), type='azure', endpoint=params['endpoint'], engine=params['engine'])
        elif self.api_type == 'ernie bot':
            response = self.llm.setv(ernie_api_key=params['api_key'], ernie_secret_key=params['secret_key'],
                          ernie_temperature=params['temperature'], ernie_top_p=params['top_p'], ernie_penalty_score=params['penalty_score'])
        elif self.api_type == 'ernie bot turbo':
            response = self.llm.setv(
                ernie_api_key=params['api_key'], ernie_secret_key=params['secret_key'], ernie_type='ernie bot turbo')
        elif self.api_type == 'chatglm api':
            response = self.llm.setv(api_key=params['api_key'], temperature=params['temperature'],
                          top_p=params['top_p'], chatglm_type=params['type'])
        elif self.api_type == 'spark api':
            response = self.llm.setv(spark_api_key=params['api_key'], spark_api_secret=params['secret_key'], spark_appid=params['appid'],
                          temperature=params['temperature'], top_k=params['top_k'], max_tokens=params['max_tokens'],spark_api_version=params['api_version'])
        elif self.api_type == 'ali api':
            response = self.llm.setv(api_key=params['api_key'], top_p=params['top_p'],
                          top_k=params['top_k'], kuake_search=params['kuake_search'])
        else:
            pass
        if response['status'] == -1:
            raise gr.Error(response['message'])

    def predict(self, input,stream=False):
        for response in self.llm.talk(input,stream):
            yield response

    def reset_state(self):
        self.llm.clear_history()
        return [], ''


class chat_base_model():
    """The overall class instantiation chat_base_api or play_base_api base on the user selection, and implemented the
        corresponding setup and inference.
    """

    def __init__(self):
        self.llm = AutoLM()
        self.model_name = None
        self.history_len = 0
        self.mysql = MySQLDB()

    def clear(self):
        if self.model_name is None:
            return [], [], '', gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(visible=False)
        elif self.model_name == 'openai' or self.model_name == 'azure openai' or self.model_name == 'ernie bot' or self.model_name == 'ernie bot turbo' or self.model_name == 'chatglm api' or self.model_name == 'spark api' or self.model_name == 'ali api':
            self.model_name = ''
            self.llm.reset_state()
            self.llm = AutoLM()
        else:
            self.model_name = ''
            if self.llm.model_name != '':
                self.llm.clear()
        return [], [], '', gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(visible=False)

    def clears(self):
        if self.model_name is None:
            return [], [], [], [],'', gr.update(value=None), gr.update(value=None),gr.update(visible=False)

        else:
            self.model_name = ''
            if self.llm.model_name != '':
                self.llm.clear()
        return [], [], [], [],'', gr.update(value=None), gr.update(value=None),gr.update(visible=False)

    def reset_state(self):
        if self.model_name == 'openai' or self.model_name == 'azure openai' or self.model_name == 'ernie bot' or self.model_name == 'ernie bot turbo' or self.model_name == 'chatglm api' or self.model_name == 'spark api' or self.model_name == 'ali api':
            self.llm.reset_state()
            return [], [], ''
        else:
            return [], [], ''
    def reset_states(self):

        return gr.update(value=[]),gr.update(value=[]),gr.update(value=[]),gr.update(value=[]),gr.update(value="")

    def load_api_params(self, params):
        self.model_name = params.get('name')
        self.llm = chat_base_api(self.model_name)
        self.llm.set_v(params)
        return '', gr.update(visible=True)

    def load_model(self, params):
        self.model_name = params.get('name')
        self.llm = AutoLM()
        self.llm.use_deepspeed = params.get('use_deepspeed')
        self.llm.load_model(max_length=params.get('max_length'), top_p=params.get('top_p'), temperature=params.get('temperature'),
                            model_name=self.model_name, use_lora=(params.get('lora') != None), lora_name=params.get('lora'), use_4bit=True if params.get('quantization') == '4 bit' else False, use_8bit=True if params.get('quantization') == '8 bit' else False)

        return '', gr.update(visible=True)

    def query_from_mysql(self, input, chatbot, database, host, user, password, port, selected_database=None, selected_table=None):
        if host == '' or user == '' or password == '' or port == '':
            raise gr.Error("请连接数据库")
        if selected_database is None:
            raise gr.Error("请选择数据库")
        self.mysql.connect(host=host, user=user,
                           password=password, port=port, database=database)
        if input == '':
            raise gr.Error('请输入内容')
        table_details = self.mysql.get_table_details()
        if table_details is None:
            raise gr.Error(f"未获取到数据库{database}的表信息")
        if hasattr(self.llm, 'llm'):
            response, sql_results, output_sql = generate_chat_responses(
                input, self.mysql, [], table_details, self.llm.llm, self.model_name)
        else:
            response, sql_results, output_sql = generate_chat_responses(user_inp=input, mysql_db=self.mysql, historical_message=[
            ], llm=self.llm, table_details=table_details, llm_name=self.model_name)
        if response['status'] == -1:
            raise gr.Error(response['message'])
        response = response['message']
        source = "\n\n"
        source += f"""<details> <summary> SQL语句</summary>\n{output_sql}\n</details>"""
        chatbot.append([input, response])
        chatbot = list(chatbot)
        chatbot[-1][-1] += source
        databses = self.mysql.get_databases()
        if selected_database not in databses:
            self.mysql.database = None
            return chatbot, '', gr.update(choices=databses, value=None), gr.update(choices=[], value=None), gr.update(visible=False)
        else:
            tables = self.mysql.get_tables(selected_database)
            if selected_table not in tables:
                return chatbot, '', gr.update(choices=databses, value=selected_database), gr.update(choices=tables, value=None), gr.update(visible=False)
            else:
                sql_results = self.mysql.get_table_data(selected_table)
                df = pd.DataFrame(sql_results)
                # 判断df的长度
                if len(sql_results) == 0:
                    fields = self.mysql.get_fields(selected_table)
                    # 将fields转成DataFrame，数据为None
                    df = pd.DataFrame(columns=fields)
                return chatbot, '', gr.update(choices=databses, value=selected_database), gr.update(choices=tables, value=selected_table), gr.update(value=df, visible=True)

    def predict(self, input, chatbot, history=[], stream=False, net=False, search='bing', search_key='', result_len='3', prompt='',doc=None,doc_type='faiss'):
        """
            Generates a response from the chatbot model given an input, mysql database and vector store if applicable.

            Args:
                self: The chat_base_model object itself.
                doc0 (str): The vector store selected.
                input (str): The input text.
                chatbot (list): The chatbot list.
                history (list): The history list.
                prompt (str): The prompt string.
                net (bool): Whether to perform internet search.
                search (str, optional): The search string. Default is None.
                search_key (str, optional): The search key string. Default is None.
                result_len (int, optional): The length of search results. Default is None.
                host (str): The host string.
                user (str): The sql username.
                password (str): The sql password string.
                port (str): The sql port string.
                type (str, optional): The vecotr store type string. Default is 'faiss'.
                selected_database (str, optional): The selected sql database string. Default is None.
                selected_table (str, optional): The selected sql table string. Default is None.

            Yields: Tuple: A tuple containing the updated chatbot list, include the newly generated response,
            history list, and an empty string.
        """
        if input == '':
            yield chatbot, history, ''
        if net and search_key == '':
            raise gr.Error("请输入search_key")
        elif net and search != '':
            internet = internet_search()
            internet.set_v(search=search, key=search_key,
                           result_len=result_len)
            answer, rep = internet.search_text(input)
            if not answer == '':
                fact_prompt = f'This following message is relative context searched from internet:\nInformation:{answer}'
                prompt = prompt+'\n'+fact_prompt
        if doc is not None:
            if doc_type == 'faiss':
                from ui.apply_knowledge import doc_qa
                if doc_qa.vector_store != None:
                    lo_doc, resp = doc_qa.get_similarity(input)
                    prompt = prompt + '\n' + lo_doc
            else:
                doc_qa = doc
                lo_doc, resp = doc_qa.get_similarity(input)
                prompt = prompt + '\n' + lo_doc
        if self.model_name == 'openai' or self.model_name == 'azure openai' or self.model_name == 'ernie bot' or self.model_name == 'ernie bot turbo' or self.model_name == 'chatglm api' or self.model_name == 'spark api' or self.model_name == 'ali api':
            if self.model_name == 'openai' or self.model_name == 'azure openai':
                self.llm.llm.charactor_prompt.content = prompt
            inputs = prompt+'\n'+input
            chatbot.append([input, ''])
            for response in self.llm.predict(inputs,stream):
                chatbot[-1][-1] += response['message']
                yield chatbot, [], ''
            if response['status'] == 0:
                if doc is not None:
                    if doc_qa.vector_store != None:
                        source = "\n\n"
                        source += "".join(
                            [f"""<details> <summary> 知识库出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}  </summary>\n"""
                            f"""{doc.page_content}\n"""
                            f"""</details>"""
                            for i, doc in enumerate(resp)])
                        chatbot[-1][-1] += source
                if net and not answer == '':
                    source = "\n\n"
                    source += "".join(
                        [f"""<details> <summary> 网络出处 [{i + 1}] <a href="{doc["link"]}" target="_blank">{doc["title"]}</a>  </summary>\n"""
                        f"""{doc['snippet']}\n"""
                        f"""</details>"""
                        for i, doc in enumerate(rep)])
                    chatbot[-1][-1] += source
                yield chatbot, [], ''
        else:
            yr = prompt+'\n'
            chatbot.append((parse_text(input), ""))
            newi = yr+input
            for response, history in self.llm._call(prompt=newi, history=history, streaming=True):
                chatbot[-1] = [parse_text(input), parse_text(response)]
                yield chatbot, history, ''
            if doc is not None:
                if doc_qa.vector_store != None:
                    source = "\n\n"
                    source += "".join(
                        [f"""<details> <summary> 知识库出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}  </summary>\n"""
                        f"""{doc.page_content}\n"""
                        f"""</details>"""
                        for i, doc in enumerate(resp)])
                    chatbot = list(chatbot)
                    chatbot[-1][-1] += source
            if net and not answer == '':
                source = "\n\n"
                source += "".join(
                    [f"""<details> <summary> 网络出处 [{i + 1}] <a href="{doc["link"]}" target="_blank">{doc["title"]}</a>  </summary>\n"""
                     f"""{doc['snippet']}\n"""
                     f"""</details>"""
                     for i, doc in enumerate(rep)])
                chatbot = list(chatbot)
                chatbot[-1][-1] += source
            yield chatbot, history, ''
            history[-1] = (history[-1][0].replace(yr, ''), history[-1][1])
    
    def paralle_api_call(self,params):
        pass

    def parallel_predict(self, input, local_chatbot1,local_chatbot2, parallel_history1=[],parallel_history2=[], net=False, search='bing', search_key='', result_len='3', prompt='',
                status=None, doc_type='faiss'):
        """
            Generates a response from the chatbot model given an input, mysql database and vector store if applicable.

            Args:
                self: The chat_base_model object itself.
                doc0 (str): The vector store selected.
                input (str): The input text.
                chatbot (list): The chatbot list.
                history (list): The history list.
                prompt (str): The prompt string.
                net (bool): Whether to perform internet search.
                search (str, optional): The search string. Default is None.
                search_key (str, optional): The search key string. Default is None.
                result_len (int, optional): The length of search results. Default is None.
                host (str): The host string.
                user (str): The sql username.
                password (str): The sql password string.
                port (str): The sql port string.
                type (str, optional): The vecotr store type string. Default is 'faiss'.
                selected_database (str, optional): The selected sql database string. Default is None.
                selected_table (str, optional): The selected sql table string. Default is None.

            Yields: Tuple: A tuple containing the updated chatbot list, include the newly generated response,
            history list, and an empty string.
        """
        prompt1 = ""
        prompt2 = ""
        if input == '':
            raise gr.Error("请输入问题")
        if net and search_key == '':
            raise gr.Error("请输入search_key")
        elif net and search != '':
            internet = internet_search()
            internet.set_v(search=search, key=search_key,
                           result_len=result_len)
            answer, rep = internet.search_text(input)
            if not answer == '':
                fact_prompt = f'This following message is relative context searched from internet:\nInformation:{answer}'
                prompt1 = prompt + '\n' + fact_prompt
                prompt2 = prompt + '\n' + fact_prompt
        chatbot1_doc = status.Chatbot1[0]
        chatbot2_doc = status.Chatbot2[0]

        if chatbot1_doc is not None:
            if doc_type == 'faiss':
                from ui.chat import parallel_local_model
                if parallel_local_model.chatbot1_vector_store != None:
                    lo_doc, resp1 = parallel_local_model.chatbot1_vector_store.get_similarity(input)
                    prompt1 = prompt1 + '\n' + lo_doc
            else:
                pass
        if chatbot2_doc is not None:
            if doc_type == 'faiss':
                from ui.chat import parallel_local_model
                if parallel_local_model.chatbot2_vector_store != None:
                    lo_doc, resp2 = parallel_local_model.chatbot2_vector_store.get_similarity(input)
                    prompt2 = prompt2 + '\n' + lo_doc
            else:
                pass


        yr1 = prompt1 + '\n'
        yr2 = prompt2 + '\n'
        local_chatbot1.append((parse_text(input), ""))
        local_chatbot2.append((parse_text(input), ""))
        newi1 = yr1 + input
        newi2 = yr2 + input
        for response, parallel_history1 in self.llm._call(prompt=newi1, history=parallel_history1, streaming=True):
            local_chatbot1[-1] = [parse_text(input), parse_text(response)]
            yield local_chatbot1,local_chatbot2, parallel_history1,parallel_history2, ''

        for response, parallel_history2 in self.llm._call(prompt=newi2, history=parallel_history2, streaming=True):
            local_chatbot2[-1] = [parse_text(input), parse_text(response)]
            yield local_chatbot1,local_chatbot2, parallel_history1,parallel_history2, ''

        if chatbot1_doc is not None:
            if parallel_local_model.chatbot1_vector_store != None:
                source = "\n\n"
                source += "".join(
                    [
                        f"""<details> <summary> 知识库出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}  </summary>\n"""
                        f"""{doc.page_content}\n"""
                        f"""</details>"""
                        for i, doc in enumerate(resp1)])
                local_chatbot1[-1][-1] += source
        if chatbot2_doc is not None:
            if parallel_local_model.chatbot2_vector_store != None:
                source = "\n\n"
                source += "".join(
                    [
                        f"""<details> <summary> 知识库出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}  </summary>\n"""
                        f"""{doc.page_content}\n"""
                        f"""</details>"""
                        for i, doc in enumerate(resp2)])
                local_chatbot2[-1][-1] += source
        if net and not answer == '':
            source = "\n\n"
            source += "".join(
                [
                    f"""<details> <summary> 网络出处 [{i + 1}] <a href="{doc["link"]}" target="_blank">{doc["title"]}</a>  </summary>\n"""
                    f"""{doc['snippet']}\n"""
                    f"""</details>"""
                    for i, doc in enumerate(rep)])
            local_chatbot1[-1][-1] += source
            local_chatbot2[-1][-1] += source
        yield local_chatbot1,local_chatbot2, parallel_history1,parallel_history2, ''
        parallel_history1[-1] = (parallel_history1[-1][0].replace(yr1, ''), parallel_history1[-1][1])
        parallel_history2[-1] = (parallel_history2[-1][0].replace(yr2, ''), parallel_history2[-1][1])


class llm_train_thread(threading.Thread):
    '''
    Thread class enables stop and restart of model trainning.
    '''

    def __init__(self, event):
        super().__init__()
        self._stop_event = event

    def stop(self):
        with open(os.path.join(real_path, '..', 'flag.txt'), 'w', encoding='utf8') as f:
            f.write('Stop')
        if not self.stopped():
            self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        tl.main()
        with open(os.path.join(real_path, '..', 'flag.txt'), 'w', encoding='utf8') as f:
            f.write('Stop')
        self._stop_event.set()


class llm_train():
    """
    Class responses to the train module on the webui page. Deal with config loading, model loading, with or without lora.
    """

    def __init__(self):
        self.event = threading.Event()
        self.llm_train_thread = llm_train_thread(self.event)
        real_path = os.path.split(os.path.realpath(__file__))[0]

        self.path = real_path
        self.accelerate_config = {
            'compute_environment': 'LOCAL_MACHINE',
            # 不使用ds的情况下直接删去此字段即可
            'deepspeed_config': {
                'deepspeed_config_file': 'deepspeed_config.json',
                # accelerate会自动检测stage，并自动配置此项，无须手动修改
                'zero3_init_flag': True
            },
            # 'DEEPSPEED'模式可同时兼容单卡多卡；
            # 在不使用deepspeed的情况下，单卡应设置为'NO'，多卡应设置为'MULTI_GPU'
            'distributed_type': 'DEEPSPEED',
            'downcast_bf16': 'no',
            'machine_rank': 0,
            'main_training_function': 'main',
            'num_machines': 1,
            # 多卡情况下需设置为要使用的显卡数量
            #
            'num_processes': 1,
            #
            #
            # 字符串类型，多卡情况下需要写成以逗号分隔的字符串，例如'0,1,2'
            'gpu_ids': '0',
            #
            # 使用deepspeed时，无需设置此项，默认为'no'
            # 候选项为'fp16', 'bf16', 'no'
            'mixed_precision': 'no',
            'rdzv_backend': 'static',
            'same_network': True,
            'tpu_env': [],
            'tpu_use_cluster': False,
            'tpu_use_sudo': False,
            # 使用cpu训练需要指定为True
            'use_cpu': False
        }
        self.deepspeed_config = {
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            # 部分以bf16训练的模型可以使用此配置，并注释掉fp16的配置
            # "bf16": {
            #     "enabled": True
            # },
            "bf16": {
                "enabled": True
            },

            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "weight_decay": "auto"
                }

            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": "auto",
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto"
                }
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "sub_group_size": 1000000000,
                "stage3_max_live_parameters": 1000000000,
                "stage3_max_reuse_distance": 1000000000,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "gradient_accumulation_steps": 8,
            "gradient_clipping": "auto",
            "steps_per_print": "auto",
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "wall_clock_breakdown": False
        }
        self.model_hyperparam_config = {"training data": "", "validation data": "",  "model name": "", "training batch size(per device)": "",
                                        "learning rate": "", "save steps": "", "max steps": ""}

    def handle_other_option(self, option):
        """
        Method response to the selection of "other" option on webui, reveal an additional input box to enable customized
        option. Deprecated in this version.
        """
        if option == "other":
            return gr.update(visible=True, interactive=True)
        else:
            return gr.update(visible=False, interactive=False)

    def handle_other_option_finished(self, function_name, state):
        """
        Function response to the selection of "other" option on webui, reveal an additional input box to enable customized
        option. Deprecated in this version.
        """
        return gr.update(choices=(state+[function_name]), value=function_name)

    def switch_CPU_GPU(self, current_device):
        """
        Function response to the selection of "GPU/CPU" option on webui, reveal an additional checkbox group to enable
        multiple gpu selection.
        """
        if current_device == "cpu":
            return gr.update(interactive=False, value=None)
        elif current_device == "gpu":
            return gr.update(interactive=True)

    @staticmethod
    def get_avaliable_gpus():
        """
        Function response to the selection of "GPU/CPU" option on webui, reveal an additional checkbox group to enable
        multiple gpu selection.
        """
        if is_available():
            device_num = device_count()
            gpu_list = [str(i) for i in range(device_num)]
            return device_num, gpu_list
        else:
            return 0, []

    def switch_checkpoint(self, lora, model):
        """
        Function return the local Lora checkpoint to webui page.
        """
        if lora is None or model is None:
            return gr.update(choices=[])
        if lora == 'Lora':
            checkpoints = []
            real_path = os.path.split(os.path.realpath(__file__))[0]
            if os.path.exists(os.path.join(real_path, "..", 'models', 'LoRA')):
                dirs = os.listdir(os.path.join(
                    real_path, "..", 'models', 'LoRA'))
            else:
                dirs = []
            for checkpoint in dirs:
                if model in checkpoint:
                    checkpoints.append(checkpoint)
            return gr.update(choices=checkpoints+[None])
        return gr.update(choices=[])

    def __load_device_config(self, LLM_device, LLM_devices):

        if LLM_device == "cpu":
            self.accelerate_config["distributed_type"] = "NO"
            self.accelerate_config["gpu_ids"] = ""  # To be decided!
            self.accelerate_config["use_cpu"] = True

        else:
            if len(LLM_devices) == 1:
                self.accelerate_config["distributed_type"] = "NO"
                self.accelerate_config["gpu_ids"] = "0"

                self.accelerate_config["use_cpu"] = False

            else:
                self.accelerate_config["distributed_type"] = "MULTI_GPU"
                # To be specified if needed
                self.accelerate_config["gpu_ids"] = LLM_devices

                self.accelerate_config["use_cpu"] = False

    def get_directories(self, path, unuse):
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d not in unuse]

    def handle_refresh_embd_and_data(self):

        real_path = os.path.split(os.path.realpath(__file__))[0]
        new_path = os.path.join(real_path, "..", "models", "Embedding")
        embs = self.get_directories(new_path, [])
        new_path = os.path.join(real_path, "..", "data",
                                'modeldata', "Embedding")
        emb_datas = self.get_directories(new_path, [])
        return gr.update(choices=embs), gr.update(choices=emb_datas)

    @staticmethod
    def detect_OS():
        """
        Method to detect current os, enable deepspeed if win.
        """
        if sys.platform.startswith('win'):
            return True
        elif sys.platform.startswith('linux'):
            return False

    def switch_deepspeed(self, deepspeed):
        """
        Method to set mixedprecision on deepspeed.
        """
        if deepspeed == "enable":
            return gr.update(choices=['no'], value="no", interactive=False)
        else:
            return gr.update(choices=['fp16', 'bf16'], value=None, interactive=True)

    def __load_llm_accelerate_config(self, deepspeed, LLM_mixed_precision, LLM_compute_environment, LLM_machine_rank, LLM_num_machines,
                                     LLM_rdzv_backend, LLM_same_network, LLM_device, LLM_devices, LLM_tpu_env, LLM_tpu_use_cluster,
                                     LLM_tpu_use_sudo, LLM_downcast_bf16, LLM_main_training_function):
        if not deepspeed:
            self.accelerate_config["mixed_precision"] = LLM_mixed_precision

        else:

            self.accelerate_config["distributed_type"] = 'DEEPSPEED'
            self.accelerate_config["mixed_precision"] = "no"

        self.__load_device_config(LLM_device, LLM_devices)
        self.accelerate_config["num_processes"] = len(
            LLM_devices) if LLM_device is not None else 0
        self.accelerate_config["same_network"] = LLM_same_network
        self.accelerate_config["compute_environment"] = LLM_compute_environment
        self.accelerate_config["downcast_bf16"] = LLM_downcast_bf16
        self.accelerate_config["machine_rank"] = LLM_machine_rank
        self.accelerate_config["main_training_function"] = LLM_main_training_function
        self.accelerate_config["num_machines"] = LLM_num_machines
        self.accelerate_config["rdzv_backend"] = LLM_rdzv_backend
        self.accelerate_config["tpu_env"] = LLM_tpu_env
        self.accelerate_config["tpu_use_cluster"] = LLM_tpu_use_cluster
        self.accelerate_config["tpu_use_sudo"] = LLM_tpu_use_sudo

    def __load_llm_deepspeed_config(self, deepspeed, LLM_mixed_precision, LLM_optimizer_type, LLM_optimizer_params_lr, LLM_optimizer_params_weight_decay,
                                    LLM_scheduler_type, LLM_scheduler_warmup_min_lr, LLM_scheduler_warmup_max_lr, LLM_scheduler_warmup_num_steps,
                                    LLM_zero_optimization_stage, LLM_zero_optimization_offload_optimizer_device,
                                    LLM_zero_optimization_pin_offload_optimizer_memory, LLM_zero_optimization_offload_param_device,
                                    LLM_zero_optimization_pin_offload_param_memory, LLM_zero_optimization_overlap_comm,
                                    LLM_zero_optimization_contiguous_gradients, LLM_zero_optimization_overlap_reduce_bucket_size,
                                    LLM_zero_optimization_stage3_prefetch_bucket_size, LLM_zero_optimization_stage3_stage3_param_persistence_threshold,
                                    LLM_zero_optimization_sub_group_size, LLM_zero_optimization_stage3_max_live_parameters,
                                    LLM_zero_optimization_stage3_max_reuse_distance, LLM_zero_optimization_stage3_gather_16bit_weights_on_model_save,
                                    LLM_gradient_accumulation_steps, LLM_gradient_clipping, LLM_steps_per_print, LLM_train_batch_size,
                                    LLM_train_micro_batch_size_per_gpu, LLM_wall_clock_breakdown,

                                    LLM_enabled, LLM_loss_scale, LLM_loss_window, LLM_initial_scale_power, LLM_hysteresis, LLM_min_loss_scale):
        if deepspeed:
            if LLM_mixed_precision == "fp16":
                del self.deepspeed_config["bf16"]
            else:
                del self.deepspeed_config["fp16"]
        if LLM_mixed_precision != "no":
            self.deepspeed_config[LLM_mixed_precision]["enabled"] = LLM_enabled
            self.deepspeed_config[LLM_mixed_precision]["loss_scale"] = LLM_loss_scale
            self.deepspeed_config[LLM_mixed_precision]["loss_scale_window"] = LLM_loss_window
            self.deepspeed_config[LLM_mixed_precision]["initial_scale_power"] = LLM_initial_scale_power
            self.deepspeed_config[LLM_mixed_precision]["hysteresis"] = LLM_hysteresis
            self.deepspeed_config[LLM_mixed_precision]["min_loss_scale"] = LLM_min_loss_scale

            self.deepspeed_config["optimizer"]["type"] = LLM_optimizer_type
            self.deepspeed_config["optimizer"]["params"]["lr"] = LLM_optimizer_params_lr
            self.deepspeed_config["optimizer"]["params"]["weight_decay"] = LLM_optimizer_params_weight_decay

            self.deepspeed_config["scheduler"]["type"] = LLM_scheduler_type
            self.deepspeed_config["scheduler"]["params"]["warmup_min_lr"] = LLM_scheduler_warmup_min_lr
            self.deepspeed_config["scheduler"]["params"]["warmup_max_lr"] = LLM_scheduler_warmup_max_lr
            self.deepspeed_config["scheduler"]["params"]["warmup_num_steps"] = LLM_scheduler_warmup_num_steps

            self.deepspeed_config["zero_optimization"]["stage"] = LLM_zero_optimization_stage
            self.deepspeed_config["zero_optimization"]["offload_optimizer"][
                "device"] = LLM_zero_optimization_offload_optimizer_device
            self.deepspeed_config["zero_optimization"]["offload_optimizer"][
                "pin_memory"] = LLM_zero_optimization_pin_offload_optimizer_memory

            self.deepspeed_config["zero_optimization"]["offload_param"]["device"] = LLM_zero_optimization_offload_param_device
            self.deepspeed_config["zero_optimization"]["offload_param"][
                "pin_memory"] = LLM_zero_optimization_pin_offload_param_memory

            self.deepspeed_config["zero_optimization"]["overlap_comm"] = LLM_zero_optimization_overlap_comm
            self.deepspeed_config["zero_optimization"]["contiguous_gradients"] = LLM_zero_optimization_contiguous_gradients
            self.deepspeed_config["zero_optimization"]["reduce_bucket_size"] = LLM_zero_optimization_overlap_reduce_bucket_size
            self.deepspeed_config["zero_optimization"]["stage3_prefetch_bucket_size"] = LLM_zero_optimization_stage3_prefetch_bucket_size
            self.deepspeed_config["zero_optimization"]["stage3_param_persistence_threshold"] = LLM_zero_optimization_stage3_stage3_param_persistence_threshold
            self.deepspeed_config["zero_optimization"]["sub_group_size"] = LLM_zero_optimization_sub_group_size
            self.deepspeed_config["zero_optimization"]["stage3_max_live_parameters"] = LLM_zero_optimization_stage3_max_live_parameters
            self.deepspeed_config["zero_optimization"]["stage3_max_reuse_distance"] = LLM_zero_optimization_stage3_max_reuse_distance
            self.deepspeed_config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = LLM_zero_optimization_stage3_gather_16bit_weights_on_model_save

            self.deepspeed_config["gradient_accumulation_steps"] = LLM_gradient_accumulation_steps
            self.deepspeed_config["gradient_clipping"] = LLM_gradient_clipping
            self.deepspeed_config["steps_per_print"] = LLM_steps_per_print
            self.deepspeed_config["train_batch_size"] = LLM_train_batch_size
            self.deepspeed_config["train_micro_batch_size_per_gpu"] = LLM_train_micro_batch_size_per_gpu
            self.deepspeed_config["wall_clock_breakdown"] = LLM_wall_clock_breakdown

    def post_train_request(self, deepspeed, LLM_mixed_precision, LLM_compute_environment, LLM_machine_rank,
                           LLM_num_machines, LLM_rdzv_backend, LLM_same_network, LLM_device, LLM_devices, LLM_tpu_env,
                           LLM_tpu_use_cluster, LLM_tpu_use_sudo, LLM_downcast_bf16, LLM_main_training_function,
                           LLM_enabled, LLM_loss_scale, LLM_loss_window, LLM_initial_scale_power, LLM_hysteresis,
                           LLM_min_loss_scale, LLM_optimizer_type, LLM_optimizer_params_lr,
                           LLM_optimizer_params_weight_decay, LLM_scheduler_type, LLM_scheduler_warmup_min_lr,
                           LLM_scheduler_warmup_max_lr, LLM_scheduler_warmup_num_steps, LLM_zero_optimization_stage,
                           LLM_zero_optimization_offload_optimizer_device,
                           LLM_zero_optimization_pin_offload_optimizer_memory,
                           LLM_zero_optimization_offload_param_device, LLM_zero_optimization_pin_offload_param_memory,
                           LLM_zero_optimization_overlap_comm, LLM_zero_optimization_contiguous_gradients,
                           LLM_zero_optimization_overlap_reduce_bucket_size,
                           LLM_zero_optimization_stage3_prefetch_bucket_size,
                           LLM_zero_optimization_stage3_stage3_param_persistence_threshold,
                           LLM_zero_optimization_sub_group_size, LLM_zero_optimization_stage3_max_live_parameters,
                           LLM_zero_optimization_stage3_max_reuse_distance,
                           LLM_zero_optimization_stage3_gather_16bit_weights_on_model_save,
                           LLM_gradient_accumulation_steps, LLM_gradient_clipping, LLM_steps_per_print,
                           LLM_train_batch_size, LLM_train_micro_batch_size_per_gpu, LLM_wall_clock_breakdown,
                           LLM_models1, LLM_data1, lora, lora_checkpoint, lora_rank, lora_use_8bit_4bit, LLM_batch_size,
                           LLM_learning_rate, LLM_save_steps, LLM_max_steps, LLM_weight_decay, LLM_epochs,
                           LLM_logging_steps):
        """
        The general method used to load config info from webui to local config file, and start the train.
        """

        if LLM_data1 is None:
            raise gr.Error('没有选择数据集')
        if os.path.exists(os.path.join(real_path, '..', 'flag.txt')):
            os.remove(os.path.join(real_path, '..', 'flag.txt'))
        self.load_llm_config(deepspeed, LLM_mixed_precision, LLM_compute_environment, LLM_machine_rank, LLM_num_machines,
                             LLM_rdzv_backend, LLM_same_network, LLM_device, LLM_devices, LLM_tpu_env, LLM_tpu_use_cluster,
                             # other option box no need to be passed since inheritated by the fomer radio
                             LLM_tpu_use_sudo, LLM_downcast_bf16, LLM_main_training_function,
                             LLM_enabled, LLM_loss_scale, LLM_loss_window, LLM_initial_scale_power, LLM_hysteresis, LLM_min_loss_scale,
                             LLM_optimizer_type, LLM_optimizer_params_lr, LLM_optimizer_params_weight_decay,
                             LLM_scheduler_type, LLM_scheduler_warmup_min_lr, LLM_scheduler_warmup_max_lr, LLM_scheduler_warmup_num_steps,
                             LLM_zero_optimization_stage, LLM_zero_optimization_offload_optimizer_device,
                             LLM_zero_optimization_pin_offload_optimizer_memory, LLM_zero_optimization_offload_param_device,
                             LLM_zero_optimization_pin_offload_param_memory, LLM_zero_optimization_overlap_comm,
                             LLM_zero_optimization_contiguous_gradients, LLM_zero_optimization_overlap_reduce_bucket_size,
                             LLM_zero_optimization_stage3_prefetch_bucket_size, LLM_zero_optimization_stage3_stage3_param_persistence_threshold,
                             LLM_zero_optimization_sub_group_size, LLM_zero_optimization_stage3_max_live_parameters,
                             LLM_zero_optimization_stage3_max_reuse_distance, LLM_zero_optimization_stage3_gather_16bit_weights_on_model_save,
                             LLM_gradient_accumulation_steps, LLM_gradient_clipping, LLM_steps_per_print, LLM_train_batch_size,
                             LLM_train_micro_batch_size_per_gpu, LLM_wall_clock_breakdown)

        self.write_accelerate_deepspeed_config()

        if lora == 'Lora':
            if lora_checkpoint is None:
                output_dir = os.path.join(
                    real_path, "..", "models", "LoRA", f"""loraed_{LLM_models1}_ft_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""")
            else:
                output_dir = os.path.join(
                    real_path, "..", "models", "LoRA", lora_checkpoint)
                lora_checkpoint = output_dir
        else:
            output_dir = os.path.join(real_path, "..", "models", "LLM",
                                      f"""llmed_{LLM_models1}_ft_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""")

        self.load_model_hyperparam(LLM_models1, LLM_data1, lora, lora_checkpoint, lora_rank, lora_use_8bit_4bit, LLM_batch_size,
                                   LLM_max_steps, LLM_save_steps, LLM_learning_rate, LLM_logging_steps, LLM_weight_decay,
                                   LLM_epochs, LLM_gradient_accumulation_steps, LLM_scheduler_type, deepspeed, output_dir)
        self.write_model_hyperparam_config()

        self.llm_train_thread.start()
        line = 0
        while True:
            if os.path.exists(os.path.join(real_path, '..', 'flag.txt')):
                if os.path.exists(f'{output_dir}/log.txt'):
                    with open(f'{output_dir}/log.txt', 'a', encoding='utf8') as f1:
                        f1.write('Done!\n')
                    with open(f'{output_dir}/log.txt', 'r', encoding='utf8') as f:
                        train_details = f.readlines()
                        if len(train_details) != line:
                            line = len(train_details)
                            yield ''.join(train_details)
                break
            if os.path.exists(f'{output_dir}/log.txt'):
                with open(f'{output_dir}/log.txt', 'r', encoding='utf8') as f:
                    train_details = f.readlines()
                    if len(train_details) != line:
                        line = len(train_details)
                        yield ''.join(train_details)
        self.event = threading.Event()
        self.llm_train_thread = llm_train_thread(self.event)

    def post_stop_request(self):
        """
        Method response to the stop train behaviour from webui, terminate the current train thread and make ready for restart.
        """
        self.llm_train_thread.stop()
        self.event = threading.Event()
        self.llm_train_thread = llm_train_thread(self.event)

    def load_llm_config(self, deepspeed, LLM_mixed_precision, LLM_compute_environment, LLM_machine_rank, LLM_num_machines,
                        LLM_rdzv_backend, LLM_same_network, LLM_device, LLM_devices, LLM_tpu_env, LLM_tpu_use_cluster,
                        # other option box no need to be passed since inheritated by the fomer radio
                        LLM_tpu_use_sudo, LLM_downcast_bf16, LLM_main_training_function,
                        LLM_enabled, LLM_loss_scale, LLM_loss_window, LLM_initial_scale_power, LLM_hysteresis, LLM_min_loss_scale,
                        LLM_optimizer_type, LLM_optimizer_params_lr, LLM_optimizer_params_weight_decay,
                        LLM_scheduler_type, LLM_scheduler_warmup_min_lr, LLM_scheduler_warmup_max_lr, LLM_scheduler_warmup_num_steps,
                        LLM_zero_optimization_stage, LLM_zero_optimization_offload_optimizer_device,
                        LLM_zero_optimization_pin_offload_optimizer_memory, LLM_zero_optimization_offload_param_device,
                        LLM_zero_optimization_pin_offload_param_memory, LLM_zero_optimization_overlap_comm,
                        LLM_zero_optimization_contiguous_gradients, LLM_zero_optimization_overlap_reduce_bucket_size,
                        LLM_zero_optimization_stage3_prefetch_bucket_size, LLM_zero_optimization_stage3_stage3_param_persistence_threshold,
                        LLM_zero_optimization_sub_group_size, LLM_zero_optimization_stage3_max_live_parameters,
                        LLM_zero_optimization_stage3_max_reuse_distance, LLM_zero_optimization_stage3_gather_16bit_weights_on_model_save,
                        LLM_gradient_accumulation_steps, LLM_gradient_clipping, LLM_steps_per_print, LLM_train_batch_size,
                        LLM_train_micro_batch_size_per_gpu, LLM_wall_clock_breakdown):

        self.__load_llm_accelerate_config(deepspeed, LLM_mixed_precision, LLM_compute_environment, LLM_machine_rank, LLM_num_machines,
                                          LLM_rdzv_backend, LLM_same_network, LLM_device, LLM_devices, LLM_tpu_env, LLM_tpu_use_cluster,
                                          LLM_tpu_use_sudo, LLM_downcast_bf16, LLM_main_training_function)

        self.__load_llm_deepspeed_config(deepspeed, LLM_mixed_precision, LLM_optimizer_type, LLM_optimizer_params_lr, LLM_optimizer_params_weight_decay,
                                         LLM_scheduler_type, LLM_scheduler_warmup_min_lr, LLM_scheduler_warmup_max_lr, LLM_scheduler_warmup_num_steps,
                                         LLM_zero_optimization_stage, LLM_zero_optimization_offload_optimizer_device,
                                         LLM_zero_optimization_pin_offload_optimizer_memory, LLM_zero_optimization_offload_param_device,
                                         LLM_zero_optimization_pin_offload_param_memory, LLM_zero_optimization_overlap_comm,
                                         LLM_zero_optimization_contiguous_gradients, LLM_zero_optimization_overlap_reduce_bucket_size,
                                         LLM_zero_optimization_stage3_prefetch_bucket_size, LLM_zero_optimization_stage3_stage3_param_persistence_threshold,
                                         LLM_zero_optimization_sub_group_size, LLM_zero_optimization_stage3_max_live_parameters,
                                         LLM_zero_optimization_stage3_max_reuse_distance, LLM_zero_optimization_stage3_gather_16bit_weights_on_model_save,
                                         LLM_gradient_accumulation_steps, LLM_gradient_clipping, LLM_steps_per_print, LLM_train_batch_size,
                                         LLM_train_micro_batch_size_per_gpu, LLM_wall_clock_breakdown,

                                         LLM_enabled, LLM_loss_scale, LLM_loss_window, LLM_initial_scale_power, LLM_hysteresis, LLM_min_loss_scale)

    def write_accelerate_deepspeed_config(self):
        """
        Method write all train config info to local.
        """
        with open(os.path.join(self.path, "..", "data", "config", "train_config", "accelerate_config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(data=self.accelerate_config,
                      stream=f, allow_unicode=True)
        with open(os.path.join(self.path, "..", "data", "config", "train_config", "deepspeed_config.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.deepspeed_config,
                    ensure_ascii=False, indent=4))
        print("Configuration Wrote")

    def load_model_hyperparam(self, LLM_models1, LLM_data1, lora, lora_checkpoint, lora_rank, lora_use_8bit_4bit, LLM_batch_size,
                              LLM_max_steps, LLM_save_steps, LLM_learning_rate, LLM_logging_steps, LLM_weight_decay,
                              LLM_train_epochs, LLM_gradient_accumulation_steps, LLM_scheduler_type, deepspeed, output_dir):
        """
        Method load all train hyperparameters from webui.
        """

        self.model_hyperparam_config["model name"] = LLM_models1
        self.model_hyperparam_config["data"] = LLM_data1
        self.model_hyperparam_config["training batch size(per device)"] = LLM_batch_size
        self.model_hyperparam_config["learning rate"] = LLM_learning_rate
        self.model_hyperparam_config["save steps"] = LLM_save_steps
        self.model_hyperparam_config["max steps"] = LLM_max_steps
        self.model_hyperparam_config["use lora"] = lora
        self.model_hyperparam_config["lora checkpoint"] = lora_checkpoint
        self.model_hyperparam_config["lora rank"] = lora_rank
        self.model_hyperparam_config["output dir"] = output_dir

        self.model_hyperparam_config["use lora 8bit 4bit"] = lora_use_8bit_4bit
        self.model_hyperparam_config["logging steps"] = LLM_logging_steps
        self.model_hyperparam_config["weight decay"] = LLM_weight_decay
        self.model_hyperparam_config["train epochs"] = LLM_train_epochs
        self.model_hyperparam_config["gradient accumulation steps"] = LLM_gradient_accumulation_steps
        if LLM_scheduler_type == "WarmupLR":
            LLM_scheduler_type = "constant_with_warmup"
        self.model_hyperparam_config["lr scheduler type"] = LLM_scheduler_type
        self.model_hyperparam_config["deepspeed"] = deepspeed

    def write_model_hyperparam_config(self):
        """
        Method writes loaded train hyperparameters to local file.
        """
        with open(os.path.join(self.path, "..", "data", "config", "train_config", "model_hyperparam_config.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.model_hyperparam_config,
                    ensure_ascii=False, indent=4))
        print("Model Configuration Wrote")

    def handle_refresh_LLM(self):
        real_path = os.path.split(os.path.realpath(__file__))[0]
        new_path = os.path.join(real_path, "..", "models", "LLM")
        models = self.get_directories(new_path, [])
        new_path = os.path.join(real_path, "..", "data", 'modeldata', "LLM")
        LLM_datas = self.get_directories(new_path, [])

        return gr.update(choices=models), gr.update(choices=LLM_datas)


class train_thread(threading.Thread):
    def __init__(self, event):
        super().__init__()
        self._stop_event = event
        self.embedding_bm = embedding_base_model()
        self.queue = queue.Queue()

    def setv(self, model_name, embed_arch, data_dir, device, batch_size, max_steps, save_steps, learning_rate, embed_logging_epochs, save_dir, epochs, weight_decay, warmup_ratio, eps, gradient_accumulation_steps):
        self.model_name = model_name
        self.embed_arch = embed_arch
        self.data_dir = data_dir
        self.device = device
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.learning_rate = learning_rate
        self.embed_logging_epochs = embed_logging_epochs
        self.save_dir = save_dir
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.eps = eps
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def stop(self):
        if hasattr(self.embedding_bm.model, 'is_stop'):
            self.embedding_bm.model.is_stop = True
        if not self.stopped():
            self.queue.put(None)
            self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def get_train_valid_file(self, data_dir):

        data_file_list = glob(f'data/modeldata/Embedding/{data_dir}/*')
        train_file = None
        valid_file = None
        for file in data_file_list:
            if 'train' in file.split('\\')[-1]:
                train_file = file
            elif 'valid' in file.split('\\')[-1]:
                valid_file = file
        return train_file, valid_file

    def get_device(self, device):
        if device == 'single_gpu':
            device = 'cuda'
        elif device == 'multi_gpu':
            pass
        return device

    def run(self):
        model_name_or_path = os.path.join(
            'models', 'Embedding', self.model_name)
        final_model = f'{self.data_dir}-{self.model_name}'
        save_dir = os.path.join('output', 'Embedding', self.save_dir if self.save_dir !=
                                '' else f'embedding_{datetime.datetime.now().strftime("%m%d_%H%M%S")}')

        device = self.get_device(self.device)
        train_file, valid_file = self.get_train_valid_file(self.data_dir)

        self.embedding_bm.clear()
        self.embedding_bm.load(
            model_name_or_path=model_name_or_path, embed_arch=self.embed_arch, device=device)
        if not self.stopped():
            for _, training_details in self.embedding_bm.train(logging_epochs=self.embed_logging_epochs, num_epochs=self.epochs, train_file=train_file, output_dir=save_dir, eval_file=valid_file, batch_size=self.batch_size, max_steps=self.max_steps, lr=self.learning_rate, weight_decay=self.weight_decay, warmup_ratio=self.warmup_ratio, eps=self.eps, gradient_accumulation_steps=self.gradient_accumulation_steps, final_model=final_model):
                self.queue.put(training_details)
            self.queue.put(None)
            self._stop_event.set()


class embedding_train_utils():
    def __init__(self) -> None:
        super().__init__()
        self.event = threading.Event()
        self.train_thread = train_thread(self.event)

    def start_train(self, model_name, embed_arch, data_dir, device, batch_size, max_steps, save_steps, learning_rate, embed_logging_epochs, save_dir, epochs, weight_decay, warmup_ratio, eps, gradient_accumulation_steps):
        if model_name is None:
            raise gr.Error('请选择模型')
        if embed_arch is None:
            raise gr.Error('请选择模型架构')
        if data_dir is None:
            raise gr.Error('请选择数据集')
        self.train_thread.setv(model_name, embed_arch, data_dir, device, batch_size, max_steps, save_steps, learning_rate,
                               embed_logging_epochs, save_dir, epochs, weight_decay, warmup_ratio, eps, gradient_accumulation_steps)
        self.train_thread.start()
        outs = ''
        while True:
            if self.train_thread.stopped():
                yield outs+'Done!' if outs != '' else ''
                break
            training_details = self.train_thread.queue.get()
            # print(training_details)
            if training_details is None:
                yield outs+'Done!' if outs != '' else ''
                break
            keys = list(training_details.keys())
            outs = ''
            for i in range(len(training_details[keys[0]])):
                if training_details[keys[3]][-1] is np.nan and training_details[keys[2]][-1] is np.nan:
                    outs += f'{keys[0]}: {training_details[keys[0]][i]}, {keys[1]}: {training_details[keys[1]][i]}\n'
                elif training_details[keys[3]][-1] is np.nan:
                    outs += f'{keys[0]}: {training_details[keys[0]][i]}, {keys[1]}: {training_details[keys[1]][i]}, {keys[2]}: {training_details[keys[2]][i]}\n'
                elif training_details[keys[2]][-1] is np.nan:
                    outs += f'{keys[0]}: {training_details[keys[0]][i]}, {keys[1]}: {training_details[keys[1]][i]}, {keys[3]}: {training_details[keys[3]][i]}\n'
                else:
                    outs += f'{keys[0]}: {training_details[keys[0]][i]}, {keys[1]}: {training_details[keys[1]][i]}, {keys[2]}: {training_details[keys[2]][i]}, {keys[3]}: {training_details[keys[3]][i]}\n'
            yield outs
        self.event = threading.Event()
        self.train_thread = train_thread(self.event)

    def stop_train(self):
        self.train_thread.stop()
        self.event = threading.Event()
        self.train_thread = train_thread(self.event)


def handle_online_tts(choice):
    if choice == "在线":
        return gr.update(choices=["中文", "英语", "日语"])
    else:
        return gr.update(choices=["中文"])


def load_javascript():
    GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse
    print("loading javascript...")
    js = '''
    <script src="file=data/config/js/custom.js"></script>
    '''

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</html>', f'{js}</html>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
