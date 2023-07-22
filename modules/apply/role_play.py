import os
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ChatMessageHistory
import json
import datetime
from modules.agent.internet_search import internet_search
from dateutil.parser import parse
from modules.model.use_api import openai_api
import random
import base64
from modules.model.llm_auto import AutoLM
import gradio as gr
from utils.local_doc import local_doc_qa


real_path = os.path.split(os.path.realpath(__file__))[0]

def image_to_base64(file_path):
    """
    Converts an image file to base64 encoded string.

    Args:
        file_path (str): The path to the image file.

    Returns:
        str: The base64 encoded string of the image.
    """
    with open(file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image

class role_play():
    """
       Role play chatbot for simulating conversations with a character. This class is the detailed implementation
        of LLM of play_base_api in ui_utils play_base_api class.

       Attributes:
           internet (internet_search): An instance of the internet_search class for internet searches.
           history (ChatMessageHistory): The chat message history.
           lo_back (local_doc_qa): An instance of the local_doc_qa class for local document search.
           lo_memory (local_doc_qa): An instance of the local_doc_qa class for local memory search.
           llm (AutoLM): An instance of the AutoLM class for language modeling.
           charactor_prompt (SystemMessage): The character prompt for the conversation.
           model_use (str): The name of the language model to use.

       Methods:
           set_llm(models0, lora0, temperature, max_tokens, top_p, openai_api_key, port, endpoint, engine):
               Sets the language model to use.

           setv(play, user_name, memory, net, search, key, result_len, back, emb, emoticon, time_c, openai_api_key, port):
               Sets the chatbot configuration.

           make_message(text: str):
               Creates a human message from the given text.

           cut_memory():
               Removes a round of conversation from the chat message history.

           message_period_to_now(message):
               Returns the number of hours between the last message and the current time.

           send_emoticon(text: str):
               Sends an emoticon based on the text input.

           talk(text):
               Generates a response based on the input text.

           clear_history():
               Clears the chat message history.
       """
    def __init__(self):
        self.internet = internet_search()
        self.history = ChatMessageHistory()
        self.lo_back = None
        self.lo_memory = None
        self.llm = None
        self.charactor_prompt = ""
        self.model_use = None
        self.play = None


    def set_llm(self,params):
        """
        Sets the language model to use.

        Args:
            models0 (str): The name of the ll model.
            lora0: The name of the lora to use.
            temperature (float): The temperature value for language generation.
            max_tokens (int): The maximum number of tokens for language generation.
            top_p (float): The top-p value for language generation.
            openai_api_key (str): The API key for OpenAI language model.
            port: The port number.
            endpoint (str): The API endpoint for the language model.
            engine (str): The engine to use for language generation.

        """
        self.model_use=params['name']
        if self.model_use == 'openai':
            self.llm=openai_api()
            self.llm.setv(openai_api_key=params['api_key'], openai_prompt=params.get('prompt',''),port=params['port'])
        elif self.model_use == 'azure openai':
            self.llm=openai_api()
            self.llm.setv(openai_api_key=params['api_key'], openai_prompt=params.get('prompt',''),type='azure',endpoint=params['endpoint'],engine=params['engine'])
        else:
            self.llm = AutoLM()
            self.llm.load_model(max_length=params['max_length'], top_p=params['top_p'],temperature=params['temperature'],model_name=self.model_use,use_lora=(params['lora'] is not None),lora_name=params['lora'],use_4bit=True if params.get('quantization') == '4 bit' else False,use_8bit=True if params.get('quantization') == '8 bit' else False)

    
    def setv(self,params):
        if self.model_use == None:
            raise gr.Error("请先选择模型")
        user_name=params.get('user_name','')
        prompt = ''
        # if params['play'] == None:
        #     prompt=''
        # else:
        self.play = params['play']
        new_path1 = os.path.join(real_path, "..","..", "data", "apply", "play",params['play'])
        with open(new_path1, "r", encoding="utf-8") as f:
            prompt = f.read()
        if self.model_use == 'openai' or self.model_use == 'azure openai':
            prompt = f'{prompt}\n User name is {user_name}，you can sometimes call me {user_name}.\n'
        
        self.time_c=params['time_c']
        if self.time_c:
            text_json = '{\
                        "msg": "The message sent by user",\
                        "time": "the system time and also your current time state, your response should consider this information"\
                    }'
            prompt = f'''{prompt}\nUser request is in JSON format, in the following format:
                    ```json
                    {text_json}
                    ```
                    Your response should be plain text, NOT IN JSON FORMAT, just response like a normal chatting.

                    Your rules:
                    You have a clear perception of time, and the system time will be provided. You need to respond accordingly, such as being more energetic during the day and feeling sleepy at night. When it's a long time since the last conversation, you should automatically deduce what happened, express your longing for me, and so on. When a period of time has passed, you will automatically deduce what happened during that period of time.
                    You should create a scenario of your own life and simulate what happens in it. If it's night, you should sleep in your room, etc.; during the day, you can do some social activities, etc. When it's time to eat, you should have something to eat. All in all, you are a real person with a real life.
                    '''
            print(prompt)
        self.charactor_prompt = SystemMessage(content=prompt)

        self.use_emoticon=params['emoticon']
        if self.use_emoticon:
            if not os.path.exists(os.path.join(real_path, "..","..", "data", "apply", "emoticon",'emoticon.json')):
                raise gr.Error("请在指定目录下放入表情包文件")
        self.net_use=params['net']

        if self.net_use:
            self.internet.set_v(search=params['search'],key=params['search_key'],result_len=params['result_len'])
        if params['background']:
            background_params = {
                'k':4,
                'score_threshold':300,
                'chunk_size':50,
                'chunk_conent':True,
                'doc':params['background']
            }
            background_params.update(params)
            self.lo_back=local_doc_qa()
            self.lo_back.load(background_params)

        if params['memory']:
            memory_params = {
                'k':6,
                'score_threshold':500,
                'chunk_size':200,
                'chunk_conent':False,
                'doc':params['memory']
            }
            memory_params.update(params)
            self.lo_memory=local_doc_qa()
            self.lo_memory.load(memory_params)
    
    def make_message(self,text: str):
        if self.time_c:
            data = {
                "msg": text,
                "time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return HumanMessage(content=json.dumps(data, ensure_ascii=False))
        else:
            return HumanMessage(content=text)

    def cut_memory(self):
        for _ in range(2):
            '''删除一轮对话'''
            if len(self.history.messages) == 0:
                raise gr.Error("prompt过长，无法生成回复，建议设置更大的Maximum length")
            first = self.history.messages.pop(0)
            print(f'删除上下文记忆: {first}')

    def message_period_to_now(self,message):
        '''返回最后一条消息到现在的小时数'''
        last_time = json.loads(message.content)['time']
        last_time = parse(last_time)
        now_time = parse(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        duration = (now_time - last_time).total_seconds() / 3600
        return duration

    def send_emoticon(self, text: str):
        '''返回 表情包 file_name'''
        probability=self.use_emoticon
        role = '''You are a system that selects and sends images based on user's text and image descriptions, and you need to send corresponding images based on the emotions expressed in the text.'''

        try:
            with open(os.path.join(real_path, "..","..", "data", "apply", "emoticon",'emoticon.json'), 'r', encoding='utf-8') as f:
                description = json.load(f)
        except:
            probability=0
        description['text'] = text
        str = json.dumps(description, ensure_ascii=False)
        prompt = f'''This is a json file, which contains the images' file_name and descriptions, you need to choose one of images. Response with a single image's file_name.\n{str}'''
        message = [
            SystemMessage(content=role),
            HumanMessage(content=prompt)
        ]

        random_number = random.random()
        if random_number <= probability:
            send = True
        else:
            send = False

        if not send:
            return ''
        else:
            if self.model_use == 'openai' or self.model_use == 'azure openai':
                reply = self.llm.get_ones_openai(message)
            else:
                replys = self.llm._call(prompt=message, history=[['','']], streaming=False)
                for reply,_ in replys:
                    pass
            for image in description["images"]:
                if image["file_name"] in reply:
                    return image["file_name"]
            return ''

    def talk(self,text):
        """
               Generates a response based on the input text.

               Args:
                   text (str): The input text.

               Yields:
                   str: The generated response.
       """
        text = text.replace("\n", " ")
        message = self.make_message(text)
        messages = [self.charactor_prompt]

        if self.net_use:
            if len(self.history.messages) >= 2: 
                q=self.history.messages[-1].content+'；'+text
            else:
                q=text
            answer,_ = self.internet.search_text(q)
            if not answer == '':
                fact_prompt = f'This following message is relative context searched from internet:\nInformation:{answer}'
                fact_message = SystemMessage(content=fact_prompt)
                messages.append(fact_message)

        if self.lo_back:
            if len(self.history.messages) >= 2: 
                q=self.history.messages[-1].content+'；'+text
            else:
                q=text
            _,an = self.lo_back.get_similarity(q)
            answer = "\n".join([doc.page_content for doc in an])
            system_message = SystemMessage(content=f'The following information is what you have experienced before:\nInformation:{answer}')
            messages.append(system_message)

        if self.lo_memory:
            if len(self.history.messages) >= 2: 
                q=self.history.messages[-1].content+'；'+text
            else:
                q=text
            _,an = self.lo_memory.get_similarity(q)
            answer = "\n".join([doc.page_content for doc in an])
            system_message = SystemMessage(content=f'The following information is what you have said before:\n{answer}\nPlease imitate the above as much as possible, and be sure to respond in a consistent tone with the above')
            messages.append(system_message)
        
        if self.time_c and len(self.history.messages) >= 2:
            duration = self.message_period_to_now(self.history.messages[-2])
            if duration > 1.0:
                system_message = SystemMessage(content=f'Passed {duration} hours since last conversation. You should simulate what you are doing during this period or make corresponding chat responses based on changes in time.')
                messages.append(system_message)

        if self.model_use == 'openai' or self.model_use == 'azure openai':
            while self.llm.llm.get_num_tokens_from_messages(messages+self.history.messages)>self.llm.max_token-500:
                self.cut_memory()
        else:
            while sum([len(i.content) for i in (messages+self.history.messages)])>self.llm.max_token-500:
                self.cut_memory()
        
        if self.model_use == 'openai' or self.model_use == 'azure openai':
            self.history.messages.append(message)
            messages.extend(self.history.messages)
            replys = self.llm.get_ones_openai(messages)
            # for reply in replys:
            #     yield reply
        else:
            for replys,history in self.llm._call(prompt=''.join([i.content for i in messages])+'\n请回复以下问题'+text, history=[[self.history.messages[i*2].content,self.history.messages[i*2+1].content] for i in range(int(len(self.history.messages)/2))], streaming=False):
                pass
            # for reply,_ in replys:
            #     yield reply
            self.history.messages.append(message)
        reply = ''.join(replys)
        self.history.add_ai_message(reply)
        emo = None
        if self.use_emoticon:
            emoticon=self.send_emoticon(reply)
            if emoticon != '':
                new_path = os.path.join(real_path, "..","..", "data", "apply", "emoticon",emoticon)
                base64_image = image_to_base64(new_path)
                emo = f'\n\n<img src="data:image/jpeg;base64,{base64_image}" alt="Local Image" >'

        return reply,emo
    
    def clear_history(self):
        self.history.clear()


