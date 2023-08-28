import gradio as gr
import requests
import json
from PIL import Image
import numpy as np
import os
import re
from datetime import datetime, timezone, timedelta
import time
from dateutil.parser import parse

real_path = os.path.dirname(os.path.realpath(__file__))

class TextImage:
    def __init__(self):
        self.message_id = None
        self.url = None
        self.sub_url = {}
        self.filename = None
        self.channel_id = None
        self.authorization = None
        self.application_id = None
        self.guild_id = None
        self.session_id = None
        self.version = None
        self.id = None
        self.flags = None
        self.header = None
        self.cur_index = 0
        self.latest_image_timestamp = datetime.now(timezone.utc) - timedelta(days=1)

    def setv(self, channel_id, authorization, application_id, guild_id, session_id, version, id, flags):
        if channel_id == "" or authorization == "" or application_id == "" or guild_id == "" or session_id == "" or version == "" or id == "" or flags == "":
            raise gr.Error("请填写完整参数")
        self.channel_id = channel_id
        self.authorization = authorization
        self.application_id = application_id
        self.guild_id = guild_id
        self.session_id = session_id
        self.version = version
        self.id = id
        self.flags = flags
        self.header = {
            'authorization': authorization
        }

    @staticmethod
    def extract_uuid(filename):
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        match = re.search(uuid_pattern, filename)
        if match:
            return match.group(0)
        else:
            return None
        
    def get_new_image(self):
        try:
            r = requests.get(
                    f'https://discord.com/api/v10/channels/{self.channel_id}/messages?limit={10}', headers=self.header)
            message_list = json.loads(r.text)
            for message in message_list:
                if (message['author']['username'] == 'Midjourney Bot') and ('**' in message['content']):
                    if len(message['attachments']) > 0:
                        if (message['attachments'][0]['filename'][-4:] == '.png') or ('(Open on website for full quality)' in message['content']):
                            message_id = message['id']
                            url = message['attachments'][0]['url']
                            filename = message['attachments'][0]['filename']
                            self.latest_image_timestamp = parse(message["timestamp"])
                            return message_id, url, filename
            return None, None, None
        except Exception as e:
            print(e)
            raise gr.Error(str(e))

    def get_whole_image(self,prompt):
        if prompt == '':
            return None
        if self.channel_id is None or self.authorization is None or self.application_id is None or self.guild_id is None or self.session_id is None or self.version is None or self.id is None or self.flags is None:
            raise gr.Error("请填写完整参数")
        payload = {
            'type': 2,
            'application_id': self.application_id,
            'guild_id': self.guild_id,
            'channel_id': self.channel_id,
            'session_id': self.session_id,
            'data': {
                'version': self.version,
                'id': self.id,
                'name': 'imagine',
                'type': 1,
                'options': [{'type': 3, 'name': 'prompt', 'value': str(prompt) + ' ' + self.flags}],
                'attachments': []}
        }
        try:
            r = requests.post('https://discord.com/api/v9/interactions',
                              json=payload, headers=self.header)
            while r.status_code != 204:
                r = requests.post(
                    'https://discord.com/api/v9/interactions', json=payload, headers=self.header)

            message_id, url, filename = self.get_new_image()
            self.message_id = message_id
            self.url = url
            self.filename = filename
            initial_image_timestamp = self.latest_image_timestamp
            max_wait_time = 300  # 最大等待时间，单位为秒
            wait_time = 0
            while wait_time < max_wait_time:
                message_id, url, filename = self.get_new_image()
                self.message_id = message_id
                self.url = url
                self.filename = filename
                current_image_timestamp = self.latest_image_timestamp
                if current_image_timestamp and current_image_timestamp > initial_image_timestamp:
                    # 发现新图片，跳出循环
                    break
                # 等待一段时间
                time.sleep(1)
                wait_time += 1
            response = requests.get(self.url)

            with open(os.path.join(real_path, '..', 'data','apply','text2image',self.filename), 'wb') as f:
                f.write(response.content)
            img = Image.open(os.path.join(real_path, '..', 'data','apply','text2image',self.filename))
            w, h = img.size
            img_array = np.array(img)
            half_width = w//2
            half_height = h//2
            img_1 = img_array[:half_height, :half_width]
            img_2 = img_array[:half_height, half_width:]
            img_3 = img_array[half_height:, :half_width]
            img_4 = img_array[half_height:, half_width:]

            # save
            filename = self.filename.split('.')[0]
            Image.fromarray(img_1).save(os.path.join(real_path, '..', 'data','apply','text2image',f'{filename}-1.png'))
            Image.fromarray(img_2).save(os.path.join(real_path, '..', 'data','apply','text2image',f'{filename}-2.png'))
            Image.fromarray(img_3).save(os.path.join(real_path, '..', 'data','apply','text2image',f'{filename}-3.png'))
            Image.fromarray(img_4).save(os.path.join(real_path, '..', 'data','apply','text2image',f'{filename}-4.png'))

            return [os.path.join(real_path, '..', 'data','apply','text2image',f'{filename}-1.png'),
                    os.path.join(real_path, '..', 'data','apply','text2image',f'{filename}-2.png'),
                    os.path.join(real_path, '..', 'data','apply','text2image',f'{filename}-3.png'),
                    os.path.join(real_path, '..', 'data','apply','text2image',f'{filename}-4.png')]
        except Exception as e:
            print(e)
            raise gr.Error(str(e))
        
    def upscale(self):
        if self.cur_index == 0:
            return None
        if self.channel_id is None or self.authorization is None or self.application_id is None or self.guild_id is None or self.session_id is None or self.version is None or self.id is None or self.flags is None:
            raise gr.Error("请填写完整参数")
        filename = self.filename.split('.')[0]
        uuid = TextImage.extract_uuid(self.filename)
        if os.path.exists(os.path.join(real_path, '..', 'data','apply','text2image',f'{filename}-{self.cur_index}-upscale.png')):
            return os.path.join(real_path, '..', 'data','apply','text2image',f'{filename}-{self.cur_index}-upscale.png')
        payload = {'type': 3,
                   'application_id': self.application_id,
                   'guild_id': self.guild_id,
                   'channel_id': self.channel_id,
                   'session_id': self.session_id,
                   "message_flags": 0,
                   "message_id": self.message_id,
                   "data": {"component_type": 2, "custom_id": f"MJ::JOB::upsample::{self.cur_index}::{uuid}"}}
        try:
            r = requests.post('https://discord.com/api/v9/interactions',
                            json=payload,
                            headers=self.header)
            while r.status_code != 204:
                r = requests.post('https://discord.com/api/v9/interactions',
                                json=payload,
                                headers=self.header)
            _, url, _ = self.get_new_image()
            self.sub_url[self.cur_index] = url
            initial_image_timestamp = self.latest_image_timestamp
            max_wait_time = 300  # 最大等待时间，单位为秒
            wait_time = 0
            while wait_time < max_wait_time:
                _, url, _ = self.get_new_image()
                self.sub_url[self.cur_index] = url
                current_image_timestamp = self.latest_image_timestamp
                if current_image_timestamp and current_image_timestamp > initial_image_timestamp:
                    # 发现新图片，跳出循环
                    break
                # 等待一段时间
                time.sleep(1)
                wait_time += 1
            response = requests.get(self.sub_url[self.cur_index])
            with open(os.path.join(real_path, '..', 'data','apply','text2image',f'{filename}-{self.cur_index}-upscale.png'), 'wb') as f:
                f.write(response.content)
            return os.path.join(real_path, '..', 'data','apply','text2image',f'{filename}-{self.cur_index}-upscale.png')

        except Exception as e:
            print(e)
            raise gr.Error(str(e))
            
        
