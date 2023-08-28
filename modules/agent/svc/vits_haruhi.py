import requests
from modules.agent.svc import inference_main
import time


def set_model_path(path):
    inference_main.set_model_path(path)


def vits_haruhi(voice_style,filename, tran,model):

    return inference_main.infer_to(voice_style, tran, filename,model)