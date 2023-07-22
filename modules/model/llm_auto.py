import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
# sys.path.append('../..')
from utils.utils import get_model_tokenizer, build_query
import deepspeed
import torch
from peft import PeftModel
from threading import Thread
from transformers import TextIteratorStreamer
from typing import List, Dict, Optional, Union
from langchain.llms.base import LLM

real_path = os.path.split(os.path.realpath(__file__))[0]
DEVICE_ = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE_ID = "0" if torch.cuda.is_available() else None
DEVICE = f"{DEVICE_}:{DEVICE_ID}" if DEVICE_ID else DEVICE_

class AutoLM(LLM):
    max_token: int = 3000
    temperature: float = 0.95
    top_p: float = 0.7
    tokenizer: object = None
    model: object = None
    history_len: int = 10
    STREAMING = True
    model_name: str = ""
    device = DEVICE_
    use_deepspeed: bool = False
    do_sample = False

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "AutoLM"

    def _call(self,
              prompt: str,
              history: List[List[str]] = [],
              streaming: bool = STREAMING):  # -> Tuple[str, List[List[str]]]:
        # history = history[-self.history_len:-1] if self.history_len > 0 else []
        if streaming:
            if "chatglm-6b" == self.model_name or "chatglm2-6b" == self.model_name:
                if self.use_deepspeed:
                    stream_chat = self.model.module.stream_chat
                else:
                    stream_chat = self.model.stream_chat
                for inum, (stream_resp, _) in enumerate(stream_chat(
                        self.tokenizer,
                        prompt,
                        history=history,
                        max_length=self.max_token,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=self.do_sample
                )):
                    if inum == 0:
                        history = history + [[prompt, stream_resp]]
                    else:
                        history[-1] = [prompt, stream_resp]
                    yield stream_resp, history
            else:
                if self.model_name == "Baichuan-13B-Chat":
                    total_input = []
                    for q, a in history:
                        q_tokens = [195] + self.tokenizer.encode(q)
                        a_tokens = [196] + self.tokenizer.encode(a) + [self.tokenizer.eos_token_id]
                        total_input += q_tokens + a_tokens
                    total_input += [195] + self.tokenizer.encode(prompt) + [196]
                    inputs = {"input_ids": torch.LongTensor([total_input[:self.max_token]]).to(self.model.device)}
                else:
                    query = build_query(
                        self.model_name, self.tokenizer, prompt, history)
                    inputs = self.tokenizer(query, max_length=self.max_token,
                                            truncation=True, return_tensors="pt").to(self.model.device)
                    if "token_type_ids" in inputs:
                        del inputs["token_type_ids"]
                streamer = TextIteratorStreamer(
                    self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                eos_token_id = self.tokenizer.eos_token_id
                if "internlm-chat-7b-8k" ==  self.model_name:
                    eos_token_id = [2, 103028]
                generation_kwargs = {**inputs, "max_length": self.max_token, "do_sample": self.do_sample, "streamer": streamer, "temperature": self.temperature, "top_p": self.top_p, "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": eos_token_id}
                thread = Thread(target=self.model.generate,
                                kwargs=generation_kwargs)
                thread.start()
                streamer_text = ""
                for inum, new_text in enumerate(streamer):
                    if "internlm-chat-7b-8k" ==  self.model_name:
                        new_text = new_text.replace("<eoa>", "")
                    streamer_text += new_text
                    if inum == 0:
                        history = history + [[prompt, streamer_text.strip()]]
                    else:
                        history[-1] = [prompt, streamer_text.strip()]
                    yield streamer_text.strip(), history
        else:
            if "chatglm-6b" == self.model_name or "chatglm2-6b" == self.model_name:
                if self.use_deepspeed:
                    chat = self.model.module.chat
                else:
                    chat = self.model.chat
                response, _ = chat(
                    self.tokenizer,
                    prompt,
                    history=history,
                    max_length=self.max_token,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample
                )
                history = history + [[prompt, response]]
                yield response, history
            else:
                if self.model_name == "Baichuan-13B-Chat":
                    total_input = []
                    for q, a in history:
                        q_tokens = [195] + self.tokenizer.encode(q) + [196]
                        a_tokens = self.tokenizer.encode(a) + [self.tokenizer.eos_token_id]
                        total_input += q_tokens + a_tokens
                    total_input += [195] + self.tokenizer.encode(prompt) + [196]
                    inputs = {"input_ids": torch.LongTensor([total_input[:self.max_token]]).to(self.model.device)}
                else:
                    query = build_query(
                        self.model_name, self.tokenizer, prompt, history)
                    inputs = self.tokenizer(query, max_length=self.max_token,
                                            truncation=True, return_tensors="pt").to(self.model.device)
                    if "token_type_ids" in inputs:
                        del inputs["token_type_ids"]
                if "internlm-chat-7b-8k" ==  self.model_name:
                    eos_token_id = [2, 103028]
                generation_kwargs = {**inputs, "max_length": self.max_token, "do_sample": self.do_sample, "temperature": self.temperature, "top_p": self.top_p, "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": eos_token_id}
                response = self.model.generate(**generation_kwargs)
                response = self.tokenizer.decode(
                    response[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
                if "internlm-chat-7b-8k" ==  self.model_name:
                    response = response.replace("<eoa>", "")
                history = history + [[prompt, response]]
                yield response, history

    def load_model(self, max_length, top_p, temperature,
                   model_name: str = "chatglm-6b",
                   llm_device=DEVICE_,
                   use_lora=False,
                   lora_name='',
                   use_8bit=False,
                   use_4bit=False,
                   device_map: Union[Optional[Dict[str, int]], str] = 'auto',
                   **kwargs):
        self.max_token = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = model_name
        self.device = llm_device
        # model_path = os.path.join("LLM", model_name)
        if not torch.cuda.is_available() or not self.device.lower().startswith("cuda"):
            use_8bit = False
            use_4bit = False
        new_path = os.path.join(real_path, "..", "..", "models", "LLM")
        original_path = os.getcwd()
        os.chdir(new_path)
        self.model, self.tokenizer = get_model_tokenizer(
            model_name, use_8bit=use_8bit, use_4bit=use_4bit, max_length=self.max_token, use_deepspeed=self.use_deepspeed, device_map=device_map)
        os.chdir(original_path)
        if use_lora:
            try:
                new_path = os.path.join(
                    real_path, "..", "..", "models", "LoRA")
                original_path = os.getcwd()
                os.chdir(new_path)
                # lora_path = os.path.join("LoRA", lora_name)
                self.model = PeftModel.from_pretrained(
                    self.model, lora_name).half()
                os.chdir(original_path)
                # 最新版本peft可直接将lora矩阵合并到原矩阵上
                if not use_8bit and not use_4bit:
                    self.model = self.model.merge_and_unload()
            except Exception as e:
                print(e)
                print("加载LoRA模型失败")

        self.model.eval()
        if self.use_deepspeed and not use_8bit and not use_4bit:
            # deepspeed init过程本身会占用一些存储，若部分模型出现OOM，可先将模型加载到内存中
            self.model = deepspeed.init_inference(self.model,
                                                  dtype=torch.half,
                                                  max_out_tokens=self.max_token,
                                                  replace_with_kernel_inject=True)

    def clear(self):
        self.model_name = ''
        del self.model
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        elif torch.backends.mps.is_available():
            try:
                from torch.mps import empty_cache
                empty_cache()
            except Exception as e:
                print(e)
                print(
                    "如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")


# if __name__ == "__main__":
#     os.chdir('../../models')
#     out = ""
#     llm = AutoLM()
#     llm.load_model(max_length=2000, top_p=0.8, temperature=0.8, model_name='internlm-chat-7b-8k', use_lora=False,
#                    lora_name='', use_4bit=True)
#     for resp, history in llm._call("你好", streaming=False):
#         print('out1:', resp)
#     for resp, history in llm._call("你好", streaming=True):
#         out = resp
#         print('out2:', out)
#     print('out3:', out)

