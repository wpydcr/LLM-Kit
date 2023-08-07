import torch
torch.cuda.device_count()
import gradio as gr
from ui import data,train,apply,chat
from utils.ui_utils import load_javascript
from utils.language_switch_utils import Localizer
import argparse

parser = argparse.ArgumentParser(description="Language")
parser.add_argument("language", type=str,default="auto", help="auto/en_UK")
arg = parser.parse_args()

localizer = Localizer(arg.language)

with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Tab(localizer("聊天")):
        chat.chat_page(localizer)
    with gr.Tab(localizer("数据")):
        data.data_page(localizer)
    with gr.Tab(localizer("应用")):
        apply.apply_page(localizer)
    with gr.Tab(localizer("训练")):
        train.train_page(localizer)

load_javascript()
demo.queue(concurrency_count=3).launch(_frontend=False,inbrowser=True)


