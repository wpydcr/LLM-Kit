import torch
torch.cuda.device_count()
import gradio as gr
from ui import data,train,apply,chat
from utils.ui_utils import load_javascript

with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Tab("chat"):
        chat.chat_page()
    with gr.Tab("data"):
        data.data_page()
    with gr.Tab("apply"):
        apply.apply_page()
    with gr.Tab("train"):
        train.train_page()

load_javascript()
demo.queue(concurrency_count=3).launch(_frontend=False,inbrowser=True)


