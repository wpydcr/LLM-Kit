# -*- coding: utf-8 -*-

import os
import json
from loguru import logger
from typing import List, Union, Optional, Any
import dataclasses

from modules.model.Embedding.text2vec.utils.base import ModelArch as Text2VecModelArch
from modules.model.Embedding.text2vec.model.bert import BertMatchModel
from modules.model.Embedding.text2vec.dataset.bert import BertMatchTrainDataset, BertMatchTestDataset
from modules.model.Embedding.text2vec.model.sentencebert import SentenceBertModel
from modules.model.Embedding.text2vec.dataset.sentencebert import SentenceBertTrainDataset, SentenceBertTestDataset
from modules.model.Embedding.text2vec.model.cosent import CosentModel
from modules.model.Embedding.text2vec.dataset.cosent import CosentTrainDataset, CosentTestDataset

real_path = os.path.split(os.path.realpath(__file__))[0]
@dataclasses.dataclass
class ModelInfo:
    module: str
    model_arch: str
    encoder_type: str = None

def load_model(
    model_name_or_path: str,
    embed_arch:str,
    device: str = None,
):
    model_arch = json.load(open(os.path.join(real_path, "..","..","..","data","config", "embedding_train", "model_arch.json"), 'r', encoding='utf-8'))
    embed_name = model_name_or_path.split(os.sep)[-1]
    if model_arch.get(embed_name, None) is None:
        model_info_dict = [model_arch[model_name] for model_name in model_arch.keys() if model_arch[model_name]["model_arch"] == embed_arch][0]
    else:
        model_info_dict = model_arch[embed_name]
    model_info = ModelInfo(**model_info_dict)
    print(model_info)
    
    if model_info.module == "TEXT2VEC":
        if model_info.model_arch == str(Text2VecModelArch.BERT):
            model = BertMatchModel(
                model_name_or_path = model_name_or_path, 
                device = device,
            )
        elif model_info.model_arch == str(Text2VecModelArch.SENTENCEBERT):
            model = SentenceBertModel(
                model_name_or_path = model_name_or_path,
                encoder_type = model_info.encoder_type,
                device = device,
            )
        elif  model_info.model_arch == str(Text2VecModelArch.COSENT):
            model = CosentModel(
                model_name_or_path = model_name_or_path,
                encoder_type = model_info.encoder_type,
                device = device,
            )
        else:
            raise ValueError(f"model_arch does not exist")
    else:
        raise ValueError(f"module does not exist")

    return model


def train_model(
    model: Any,
    train_file: str,
    output_dir: str,
    eval_file: str = None,
    verbose: bool = True,
    batch_size: int = 32,
    num_epochs: int = 1,
    weight_decay: float = 0.01,
    seed: int = 42,
    warmup_ratio: float = 0.1,
    lr: float = 2e-5,
    eps: float = 1e-6,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    max_steps: int = -1,
    logging_epochs: int = 1,
    final_model=None
):
    if isinstance(model, BertMatchModel):
        train_dataset = BertMatchTrainDataset(
            model.tokenizer, BertMatchTrainDataset.load_train_data(train_file), model.max_seq_length)
        eval_dataset = BertMatchTestDataset(
            model.tokenizer, BertMatchTestDataset.load_test_data(eval_file), model.max_seq_length) if eval_file else None
    elif isinstance(model, SentenceBertModel):
        train_dataset = SentenceBertTrainDataset(
            model.tokenizer, SentenceBertTrainDataset.load_train_data(train_file), model.max_seq_length)
        eval_dataset = SentenceBertTestDataset(
            model.tokenizer, SentenceBertTestDataset.load_test_data(eval_file), model.max_seq_length) if eval_file else None
    elif isinstance(model, CosentModel):
        train_dataset = CosentTrainDataset(
            model.tokenizer, CosentTrainDataset.load_train_data(train_file), model.max_seq_length)
        eval_dataset = CosentTestDataset(
            model.tokenizer, CosentTestDataset.load_test_data(eval_file), model.max_seq_length) if eval_file else None

    for global_step, training_details in model.train(
        train_dataset,
        output_dir,
        eval_dataset=eval_dataset,
        verbose=verbose,
        batch_size=batch_size,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        seed=seed,
        warmup_ratio=warmup_ratio,
        lr=lr,
        eps=eps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        final_model=final_model,
        logging_epochs=logging_epochs,
    ):
        yield global_step, training_details
    
    logger.info(f"Training model done. Saved to {output_dir}.")
    # return global_step, training_details


def eval_model(
    model: Any,
    eval_file: str,
    output_dir: str = None,
    verbose: bool = True,
    batch_size: int = 16,
):
    if isinstance(model, BertMatchModel):
        eval_dataset = BertMatchTestDataset(
            model.tokenizer, BertMatchTestDataset.load_test_data(eval_file), model.max_seq_length)
    elif isinstance(model, SentenceBertModel):
        eval_dataset = SentenceBertTestDataset(
            model.tokenizer, SentenceBertTestDataset.load_test_data(eval_file), model.max_seq_length)
    elif isinstance(model, CosentModel):
        eval_dataset = CosentTestDataset(
            model.tokenizer, CosentTestDataset.load_test_data(eval_file), model.max_seq_length)

    result = model.eval_model(
        eval_dataset,
        output_dir,
        verbose,
        batch_size,
    )
    
    return result


if __name__ == "__main__":
    real_path = os.path.realpath(__file__)
    base_model_relative_path: str = '../../../../models/Embedding/'
    base_data_relative_path: str = '../../../../data/modeldata/Embedding/'
    base_output_relative_path: str = '../../../../outputs/Embedding/'
    
    # model_name_or_path = os.path.abspath(os.path.join(real_path, base_model_relative_path, "bert-base-uncased"))
    # model_name_or_path = os.path.abspath(os.path.join(real_path, base_model_relative_path, "paraphrase-multilingual-MiniLM-L12-v2"))
    model_name_or_path = os.path.abspath(os.path.join(real_path, base_model_relative_path, "text2vec-base-chinese"))
    train_file = os.path.abspath(os.path.join(real_path, base_data_relative_path, "STS-B/STS-B.train.data"))
    eval_file = os.path.abspath(os.path.join(real_path, base_data_relative_path, "STS-B/STS-B.valid.data"))
    output_dir = os.path.abspath(os.path.join(real_path, base_output_relative_path, "X-STSB"))
    
    print(model_name_or_path)
    m = load_model(model_name_or_path)

    train_model(
        m,
        train_file,
        output_dir,
        eval_file,
    )
    
    
    
    
    