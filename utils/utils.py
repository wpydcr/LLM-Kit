from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    BitsAndBytesConfig
)
from transformers.generation.utils import GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model
import shutil
import os
import re
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
try:
    from xformers import ops as xops
    pad_to_multiple_of_8 = True
except:
    pad_to_multiple_of_8 = False

real_path = os.path.split(os.path.realpath(__file__))[0]

def parse_input_string(input_string):
    input_string = re.sub(r'<br>', '。', input_string)
    # 过滤掉所有的html标签
    cleaned_string = re.sub(r'<[^>]+>', '', input_string)

    return cleaned_string

def get_model_name(path):
    while True:
        parent, name = os.path.split(path)
        if name != "":
            return name
        if parent == "":
            raise Exception("Can't get model name.")
        path = parent

def copy_custom_files(source, target):
    model_name = get_model_name(source)
    if (
        "chatglm-6b" == model_name or 
        "chatglm2-6b" == model_name or 
        "chatglm2-6b-32k" == model_name
    ):
        shutil.copy(os.path.join(source, "configuration_chatglm.py"), target)
        shutil.copy(os.path.join(source, "modeling_chatglm.py"), target)
        shutil.copy(os.path.join(source, "quantization.py"), target)
        shutil.copy(os.path.join(source, "tokenization_chatglm.py"), target)
    elif (
        "phoenix-inst-chat-7b" == model_name or
        "phoenix-inst-chat-7b-v1.1" == model_name or
        "Guanaco" == model_name or
        "baichuan-vicuna-chinese-7b" == model_name or
        "chinese-alpaca-2-7b" == model_name
    ):
        pass
    elif "moss-moon-003-sft" == model_name:
        shutil.copy(os.path.join(source, "configuration_moss.py"), target)
        shutil.copy(os.path.join(source, "modeling_moss.py"), target)
        shutil.copy(os.path.join(source, "tokenization_moss.py"), target)
    elif (
        "Baichuan-13B-Chat" == model_name or
        "Baichuan2-13B-Chat" == model_name or
        "Baichuan2-7B-Chat" == model_name
        ):
        shutil.copy(os.path.join(source, "configuration_baichuan.py"), target)
        shutil.copy(os.path.join(source, "modeling_baichuan.py"), target)
        shutil.copy(os.path.join(source, "quantizer.py"), target)
        shutil.copy(os.path.join(source, "tokenization_baichuan.py"), target)
        shutil.copy(os.path.join(source, "generation_utils.py"), target)
    elif "internlm-chat-7b-8k" == model_name:
        shutil.copy(os.path.join(source, "configuration_internlm.py"), target)
        shutil.copy(os.path.join(source, "modeling_internlm.py"), target)
        shutil.copy(os.path.join(source, "tokenization_internlm.py"), target)
    elif (
        "Qwen-7B-Chat" == model_name or
        "Qwen-14B-Chat" == model_name
    ):
        shutil.copy(os.path.join(source, "configuration_qwen.py"), target)
        shutil.copy(os.path.join(source, "modeling_qwen.py"), target)
        shutil.copy(os.path.join(source, "tokenization_qwen.py"), target)
        shutil.copy(os.path.join(source, "qwen_generation_utils.py"), target)
        shutil.copy(os.path.join(source, "cache_autogptq_cuda_256.cpp"), target)
        shutil.copy(os.path.join(source, "cache_autogptq_cuda_kernel_256.cu"), target)
        shutil.copy(os.path.join(source, "cpp_kernels.py"), target)
    else:
        raise NotImplementedError("Model is not implemented.")

def get_model_tokenizer(path, use_8bit=False, use_4bit=False, max_length=1024, use_deepspeed=False, device_map=None, dtype='fp16'):
    if dtype == 'fp16':
        data_type = torch.float16
    elif dtype == 'bf16':
        data_type = torch.bfloat16
    else:
        data_type = torch.float32
    model_name = get_model_name(path)
    new_path = os.path.join(real_path, "..", "models", "LLM")
    original_path = os.getcwd()
    os.chdir(new_path)
    quantization_config = None
    if use_deepspeed:
        device_map = None
        use_8bit = False
        use_4bit = False
    if (use_8bit or use_4bit) and device_map is None:
        device_map = "auto"
    if use_4bit:
        use_8bit = False
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=data_type,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if "chatglm-6b-int4" == model_name:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, cache_dir='./')
        if max_length > config.max_sequence_length:
            config.update({"max_sequence_length": max_length})
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, cache_dir='./')
        model = AutoModel.from_pretrained(
            path, config=config, cache_dir='./',
            trust_remote_code=True, 
            # empty_init=False,
            load_in_8bit=use_8bit,
            device_map=device_map,
            torch_dtype=data_type)
    elif "chatglm-6b" == model_name:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, cache_dir='./')
        if max_length > config.max_sequence_length:
            config.update({"max_sequence_length": max_length})
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, cache_dir='./')
        model = AutoModel.from_pretrained(
            path, config=config, cache_dir='./',
            trust_remote_code=True,
            empty_init=False,
            load_in_8bit=use_8bit,
            device_map=device_map,
            quantization_config=quantization_config, 
            torch_dtype=data_type)
    elif "chatglm2-6b" == model_name or "chatglm2-6b-32k" == model_name:
        config = AutoConfig.from_pretrained(path, cache_dir='./', trust_remote_code=True)
        if max_length > config.seq_length:
            config.update({"seq_length": max_length})
        tokenizer = AutoTokenizer.from_pretrained(
            path, cache_dir='./', trust_remote_code=True)
        model = AutoModel.from_pretrained(
            path, config=config, cache_dir='./',
            trust_remote_code=True, 
            empty_init=False,
            load_in_8bit=use_8bit,
            device_map=device_map,
            quantization_config=quantization_config, 
            torch_dtype=data_type)
    elif "phoenix-inst-chat-7b" == model_name or "phoenix-inst-chat-7b-v1.1" == model_name:
        config = AutoConfig.from_pretrained(path, cache_dir='./')
        if max_length > config.seq_length:
            config.update({"seq_length": max_length})
        tokenizer = AutoTokenizer.from_pretrained(path, cache_dir='./')
        model = AutoModelForCausalLM.from_pretrained(
            path, config=config,cache_dir='./',
            load_in_8bit=use_8bit,
            device_map=device_map,
            quantization_config=quantization_config,  
            torch_dtype=data_type)
    elif "moss-moon-003-sft" == model_name:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, cache_dir='./')
        if max_length > config.n_positions:
            config.update({"n_positions": max_length})
            config.update({"max_position_embeddings": max_length})
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, cache_dir='./')
        model = AutoModelForCausalLM.from_pretrained(
            path, config=config, cache_dir='./',
            trust_remote_code=True, 
            load_in_8bit=use_8bit, 
            device_map=device_map,
            quantization_config=quantization_config, 
            torch_dtype=data_type)
        # moss没有用到pad_token，为与其他模型保持一致，进行额外增加
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif "Guanaco" == model_name:
        config = AutoConfig.from_pretrained(path, cache_dir='./')
        if max_length > config.max_position_embeddings:
            config.update({"max_position_embeddings": max_length})
        tokenizer = AutoTokenizer.from_pretrained(path, cache_dir='./',use_fast=False)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        model = AutoModelForCausalLM.from_pretrained(
            path, config=config, cache_dir='./',
            load_in_8bit=use_8bit,
            device_map=device_map,
            quantization_config=quantization_config,  
            torch_dtype=data_type,
            )
    elif "baichuan-vicuna-chinese-7b" == model_name:
        config = AutoConfig.from_pretrained(path, cache_dir='./')
        if max_length > config.max_position_embeddings:
            config.update({"max_position_embeddings": max_length})
            config.update({"max_sequence_length": max_length})
        tokenizer = AutoTokenizer.from_pretrained(
            path, cache_dir='./', use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            path, config=config, cache_dir='./',
            load_in_8bit=use_8bit, 
            device_map=device_map,
            quantization_config=quantization_config, 
            torch_dtype=data_type,
        )
        tokenizer.pad_token_id = model.config.pad_token_id
    elif (
        "Baichuan-13B-Chat" == model_name or 
        "Baichuan2-13B-Chat" == model_name or
        "Baichuan2-7B-Chat" == model_name
    ):
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, cache_dir="./")
        if max_length > config.model_max_length:
            config.update({"model_max_length": max_length})
        tokenizer = AutoTokenizer.from_pretrained(
            path, use_fast=False, trust_remote_code=True, cache_dir="./"
            )
        model = AutoModelForCausalLM.from_pretrained(
            path, config=config, cache_dir='./',
            trust_remote_code=True,
            load_in_8bit=use_8bit,
            device_map=device_map,
            quantization_config=quantization_config,  
            torch_dtype=data_type,
            )
        model.generation_config = GenerationConfig.from_pretrained(path, trust_remote_code=True, cache_dir="./")
        tokenizer.user_token_id = model.generation_config.user_token_id
        tokenizer.user_token = tokenizer.convert_ids_to_tokens(tokenizer.user_token_id)
        tokenizer.assistant_token_id = model.generation_config.assistant_token_id
        tokenizer.assistant_token = tokenizer.convert_ids_to_tokens(tokenizer.assistant_token_id)
    elif "internlm-chat-7b-8k" == model_name:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, cache_dir="./")
        if max_length > config.max_position_embeddings:
            config.update({"max_position_embeddings": max_length})
        tokenizer = AutoTokenizer.from_pretrained(
            path, use_fast=False, trust_remote_code=True, cache_dir="./"
            )
        model = AutoModelForCausalLM.from_pretrained(
            path, config=config, cache_dir='./',
            trust_remote_code=True,
            load_in_8bit=use_8bit,
            device_map=device_map,
            quantization_config=quantization_config,  
            torch_dtype=data_type,
            )
    elif "chinese-alpaca-2-7b" == model_name:
        config = AutoConfig.from_pretrained(path, cache_dir="./")
        if max_length > config.max_position_embeddings:
            config.update({"max_position_embeddings": max_length})
        tokenizer = AutoTokenizer.from_pretrained(path, cache_dir="./", use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            path, config=config, cache_dir="./",
            load_in_8bit=use_8bit,
            device_map=device_map,
            quantization_config=quantization_config,  
            torch_dtype=data_type,
            )
    elif (
        "Qwen-7B-Chat" == model_name or
        "Qwen-14B-Chat" == model_name
    ):
        config = AutoConfig.from_pretrained(path, cache_dir="./", trust_remote_code=True)
        if dtype == 'fp16':
            config.fp16 = True
            config.bf16 = False
            config.fp32 = False
        elif dtype == 'bf16':
            config.fp16 = False
            config.bf16 = True
            config.fp32 = False
        else:
            config.fp16 = False
            config.bf16 = False
            config.fp32 = True
        if use_4bit or use_8bit:
            config.update({"use_flash_attn": False})
        if max_length > config.max_position_embeddings:
            config.update({"max_position_embeddings": max_length})
        tokenizer = AutoTokenizer.from_pretrained(
            path, cache_dir="./", trust_remote_code=True, 
            pad_token="<|im_end|>", eos_token="<|im_end|>"
        )
        model = AutoModelForCausalLM.from_pretrained(
            path, config=config, device_map=device_map, load_in_8bit=use_8bit, 
            cache_dir="./", torch_dtype=data_type, trust_remote_code=True,
            quantization_config=quantization_config,
        )
        model.generation_config = GenerationConfig.from_pretrained(path, trust_remote_code=True, cache_dir="./")
    else:
        raise NotImplementedError("Model is not implemented.")
    os.chdir(original_path)
    return model, tokenizer

def get_lora_model(model, path, lora_rank, checkpoint):
    model_name = get_model_name(path)
    if (
        "chatglm-6b" == model_name or
        "chatglm2-6b-32k" == model_name or
        "phoenix-inst-chat-7b" == model_name or
        "phoenix-inst-chat-7b-v1.1" == model_name or
        "Guanaco" == model_name or
        "baichuan-vicuna-chinese-7b" == model_name or
        "chinese-alpaca-2-7b" == model_name
    ):
        target_modules = None
    elif "moss-moon-003-sft" == model_name:
        target_modules = ["qkv_proj"]
    elif (
        "Baichuan-13B-Chat" == model_name or
        "Baichuan2-13B-Chat" == model_name or
        "Baichuan2-7B-Chat" == model_name
    ):
        target_modules = ["W_pack"]
    elif "internlm-chat-7b-8k" == model_name:
        target_modules = ["q_proj", "k_proj"]
    elif (
        "Qwen-7B-Chat" == model_name or
        "Qwen-14B-Chat" == model_name
    ):
        target_modules = ["c_attn"]
    else:
        raise NotImplementedError("Model is not implemented.")
    peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules
        )
    model = get_peft_model(model, peft_config)
    if checkpoint is not None:
        model = PeftModel.from_pretrained(model, checkpoint)
        for name, p in model.named_parameters():
            if "lora" in name:
                p.requires_grad = True
    model.print_trainable_parameters()
    return model


def get_preprocess_datacollator(path):
    model_name = get_model_name(path)
    if "chatglm-6b" == model_name:
        return preprocess_4_chatglm, data_collator_4_chatglm
    elif "chatglm2-6b" == model_name or "chatglm2-6b-32k" == model_name:
        return preprocess_4_chatglm2, data_collator_4_chatglm2
    elif "phoenix-inst-chat-7b" == model_name or "phoenix-inst-chat-7b-v1.1" == model_name:
        return preprocess_4_phoenix, data_collator_4_phoenix
    elif "moss-moon-003-sft" == model_name:
        return preprocess_4_moss, data_collator_4_moss
    elif "Guanaco" == model_name:
        return preprocess_4_gunaco, data_collator_4_guanaco
    elif "baichuan-vicuna-chinese-7b" == model_name:
        return preprocess_4_baichuan, data_collator_4_baichuan
    elif (
        "Baichuan-13B-Chat" == model_name or
        "Baichuan2-13B-Chat" == model_name or
        "Baichuan2-7B-Chat" == model_name
    ):
        return preprocess_4_baichuan_13b_chat, data_collator_4_baichuan_13b_chat
    elif "internlm-chat-7b-8k" == model_name:
        return preprocess_4_internlm, data_collator_4_internlm
    elif "chinese-alpaca-2-7b" == model_name:
        return preprocess_4_chinese_alpaca_2, data_collator_4_chinese_alpaca_2
    elif (
        "Qwen-7B-Chat" == model_name or
        "Qwen-14B-Chat" == model_name
    ):
        return preprocess_4_qwen, data_collator_4_qwen
    else:
        raise NotImplementedError("Model is not implemented.")

def build_query(path, tokenizer, question, history):
    model_name = get_model_name(path)
    query = ""
    if "phoenix-inst-chat-7b" == model_name or "phoenix-inst-chat-7b-v1.1" == model_name:
        for q, a in history:
            query += "Human: {}{}Assistant: {}{}{}".format(q, tokenizer.eos_token, tokenizer.bos_token, a, tokenizer.eos_token)
        query += "Human: {}{}Assistant:{}".format(question, tokenizer.eos_token, tokenizer.bos_token)
    elif "moss-moon-003-sft" == model_name:
        for q, a in history:
            query += "<|Human|>: {}<eoh>\n<|MOSS|>: {}{}\n".format(q, a, tokenizer.eos_token)
        query += "<|Human|>: {}<eoh>\n<|MOSS|>:".format(question)
    elif "Guanaco" == model_name:
        for q, a in history:
            query += "### Instruction: \n{}\n\n### Response: \n{}{}\n".format(q, a, tokenizer.eos_token)
        query += "### Instruction: \n{}\n\n### Response:".format(question)
    elif "baichuan-vicuna-chinese-7b" == model_name:
        for q, a in history:
            query += "USER: {} ASSISTANT: {}{}".format(q, a, tokenizer.eos_token)
        query += "USER: {} ASSISTANT:".format(question)
    elif "internlm-chat-7b-8k" == model_name:
        for num, (q, a) in enumerate(history):
            start_token = "" if num == 0 else tokenizer.bos_token
            query += "{}<|User|>:{}<eoh>\n<|Bot|>:{}<eoa>\n".format(start_token, q, a)
        start_token = "" if query == "" else tokenizer.bos_token
        query += "{}<|User|>:{}<eoh>\n<|Bot|>:".format(start_token, question)
    elif "chinese-alpaca-2-7b" == model_name:
        for num, (q, a) in enumerate(history):
            start_token = "" if num == 0 else tokenizer.bos_token
            query += "{}[INST] {} [/INST] {}{}".format(start_token, q, a, tokenizer.eos_token)
        start_token = "" if query == "" else tokenizer.bos_token
        query += "{}[INST] {} [/INST]".format(start_token, question)
    elif "chatglm2-6b-32k" == model_name:
        round = 1
        for q, a in history:
            query += "[Round {}]\n\n问：{}\n\n答：{}{}".format(round, q, a, tokenizer.eos_token)
            round += 1
        query += "[Round {}]\n\n问：{}\n\n答：".format(round, question)
    elif (
        "Qwen-7B-Chat" == model_name or
        "Qwen-14B-Chat" == model_name
    ):
        query += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        for q, a in history:
            query += "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>\n".format(q, a)
        query += "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(question)
    elif (
        "Baichuan-13B-Chat" == model_name or
        "Baichuan2-13B-Chat" == model_name or
        "Baichuan2-7B-Chat" == model_name
    ):
        for q, a in history:
            query += tokenizer.user_token + q + tokenizer.assistant_token + a + tokenizer.eos_token
        query += tokenizer.user_token + question + tokenizer.assistant_token
    else:
        raise NotImplementedError("Model is not implemented.")
    return query
 
def build_tokens(full_prompt, user_prompt, tokenizer, max_length):
    tokenized_full_prompt = tokenizer(full_prompt, max_length=max_length, truncation=True)
    if (
        tokenized_full_prompt["input_ids"][-1] != tokenizer.eos_token_id
        and len(tokenized_full_prompt["input_ids"]) < max_length
    ):
        tokenized_full_prompt["input_ids"].append(tokenizer.eos_token_id)
        tokenized_full_prompt["attention_mask"].append(1)
    tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()
    tokenized_user_prompt = tokenizer(user_prompt, max_length=max_length, truncation=True)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    if tokenized_user_prompt["input_ids"][-1] == tokenizer.eos_token_id:
        user_prompt_len -= 1
    tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
    return tokenized_full_prompt

# --------------- Phoenix ---------------
def preprocess_4_phoenix(example, tokenizer, max_length):
    question = example["question"]
    answer = example["answer"]
    text = "Human: {}{}Assistant: {}{}{}".format(
        question, tokenizer.eos_token, tokenizer.bos_token, answer, tokenizer.eos_token)
    inputs = tokenizer(text, max_length=max_length, truncation=True)
    sep_idx = inputs["input_ids"].index(tokenizer.bos_token_id)
    inputs["labels"] = [-100] * sep_idx + inputs["input_ids"][sep_idx:]
    return inputs


def data_collator_4_phoenix(features, tokenizer):
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = (max(len_ids) + 7) // 8 * 8 if pad_to_multiple_of_8 else max(len_ids)
    input_ids = []
    attention_mask = []
    labels_list = []
    for ids_l, feature in zip(len_ids, features):
        ids = feature["input_ids"]
        att = feature["attention_mask"]
        labels = feature["labels"]
        labels += [-100] * (longest - ids_l)
        ids += [tokenizer.pad_token_id] * (longest - ids_l)
        att += [0] * (longest - ids_l)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(torch.LongTensor(ids))
        attention_mask.append(torch.LongTensor(att))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels_list)
    }

# --------------- ChatGLM ---------------
def preprocess_4_chatglm(example, tokenizer, max_length):
    prompt = example["question"]
    target = example["answer"]
    prompt_ids = tokenizer.encode(
        prompt, max_length=max_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    return {"input_ids": input_ids[:max_length], "seq_len": len(prompt_ids)}


def data_collator_4_chatglm(features, tokenizer):
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = (max(len_ids) + 7) // 8 * 8 if pad_to_multiple_of_8 else max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in zip(len_ids, features):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1):] +
            [-100] * (longest - ids_l)
        )
        ids += [tokenizer.pad_token_id] * (longest - ids_l)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(torch.LongTensor(ids))
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

# --------------- MOSS ---------------
def preprocess_4_moss(example, tokenizer, max_length):
    question = example["question"]
    answer = example["answer"]
    user_prompt = "<|Human|>: {}<eoh>\n<|MOSS|>: ".format(question)
    full_prompt = user_prompt + answer
    return build_tokens(full_prompt, user_prompt, tokenizer, max_length)

data_collator_4_moss = data_collator_4_phoenix

# --------------- Guanaco ---------------
def preprocess_4_gunaco(example, tokenizer, max_length):
    question = example["question"]
    answer = example["answer"]
    user_prompt = "### Instruction: \n{}\n\n### Response: \n".format(question)
    full_prompt = user_prompt + answer
    return build_tokens(full_prompt, user_prompt, tokenizer, max_length)

data_collator_4_guanaco = data_collator_4_phoenix

# --------------- baichuan ---------------
def preprocess_4_baichuan(example, tokenizer, max_length):
    question = example["question"]
    answer = example["answer"]
    user_prompt = "USER: {} Assistant: ".format(question)
    full_prompt = user_prompt + answer
    return build_tokens(full_prompt, user_prompt, tokenizer, max_length)

data_collator_4_baichuan = data_collator_4_phoenix

# --------------- ChatGLM2 ---------------
def preprocess_4_chatglm2(example, tokenizer, max_length):
    prompt = "[Round 1]\n\n问：{}\n\n答：".format(example["question"])
    target = example["answer"]
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(
        target,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    position_ids = list(range(len(input_ids)))
    return {"input_ids": input_ids[:max_length], 
            "seq_len": len(prompt_ids),
            "position_ids": position_ids[:max_length]
            }

def data_collator_4_chatglm2(features, tokenizer):
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = (max(len_ids) + 7) // 8 * 8 if pad_to_multiple_of_8 else max(len_ids)
    input_ids = []
    labels_list = []
    pos_list = []
    for ids_l, feature in zip(len_ids, features):
        ids = feature["input_ids"]
        pos = feature["position_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1):] +
            [-100] * (longest - ids_l)
        )
        ids += [tokenizer.pad_token_id] * (longest - ids_l)
        pos += [0] * (longest - ids_l) 
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(torch.LongTensor(ids))
        pos_list.append(torch.LongTensor(pos))
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    position_ids = torch.stack(pos_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids
    }


# --------------- Baichuan-13B-Chat ---------------
def preprocess_4_baichuan_13b_chat(example, tokenizer, max_length):
    question = example["question"]
    answer = example["answer"]
    question_tokens = tokenizer.encode(question)
    answer_tokens = tokenizer.encode(answer)
    full_tokens = [tokenizer.user_token_id] + question_tokens + [tokenizer.assistant_token_id] + answer_tokens + [tokenizer.eos_token_id]
    user_prompt_len = len(question_tokens) + 2
    labels = [-100] * user_prompt_len + full_tokens[user_prompt_len:]
    full_tokens = full_tokens[:max_length]
    labels = labels[:max_length]
    attention_mask = [1] * len(full_tokens)
    return {
        "input_ids": full_tokens,
        "attention_mask": attention_mask,
        "labels": labels
    }

data_collator_4_baichuan_13b_chat = data_collator_4_phoenix 


# --------------- internlm ---------------
def preprocess_4_internlm(example, tokenizer, max_length):
    question = example["question"]
    answer = example["answer"]
    user_prompt = """<s><|User|>:{}<eoh>\n<|Bot|>:""".format(question)
    prompt_tokens = tokenizer.encode(user_prompt)
    full_prompt = user_prompt + answer + "<eoa>"
    full_tokens = tokenizer.encode(full_prompt)
    user_prompt_len = len(prompt_tokens)
    labels = [-100] * user_prompt_len + full_tokens[user_prompt_len:]
    full_tokens = full_tokens[:max_length]
    labels = labels[:max_length]
    attention_mask = [1] * len(full_tokens)
    return {
        "input_ids": full_tokens,
        "attention_mask": attention_mask,
        "labels": labels
    }

data_collator_4_internlm = data_collator_4_phoenix


# --------------- chinese-alpaca-2 ---------------
def preprocess_4_chinese_alpaca_2(example, tokenizer, max_length):
    question = example["question"]
    answer = example["answer"]
    user_prompt = """[INST] {} [/INST] """.format(question)
    full_prompt = user_prompt + answer
    return build_tokens(full_prompt, user_prompt, tokenizer, max_length)

data_collator_4_chinese_alpaca_2 = data_collator_4_phoenix


# --------------- Qwen ---------------
def preprocess_4_qwen(example, tokenizer, max_length):
    question = example["question"]
    answer = example["answer"]
    user_prompt = """<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n""".format(question)
    full_prompt = user_prompt + answer
    return build_tokens(full_prompt, user_prompt, tokenizer, max_length)

data_collator_4_qwen= data_collator_4_phoenix
