# -*- coding: utf-8 -*-

import os
import json
from loguru import logger
import numpy as np
from tqdm.auto import tqdm, trange
from typing import List, Union, Optional, Any
import math
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from modules.model.Embedding.text2vec.utils.base import compute_pearsonr, compute_spearmanr, set_seed, ModelArch, EncoderType


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"


class BaseModel:
    def __init__(
        self,
        model_arch: Union[str, ModelArch] = None,
        model_name_or_path: str = None,
        num_classes: int = 2,
        max_seq_length: int = 128,
        encoder_type: Union[str, EncoderType] = None,
        device: Optional[str] = None,
        model: Any = None,
        tokenizer: Any = None,
    ):
        model_arch = ModelArch.from_string(model_arch) if isinstance(
            model_arch, str) else model_arch
        if model_arch not in list(ModelArch):
            raise ValueError(f"model_arch must be in {list(ModelArch)}")
        self.model_arch = model_arch

        encoder_type = EncoderType.from_string(encoder_type) if isinstance(
            encoder_type, str) else encoder_type
        if encoder_type not in list(EncoderType) and encoder_type is not None:
            raise ValueError(
                f"encoder_type must be in {list(EncoderType)} or {None}")
        self.encoder_type = encoder_type

        if device is None:
            device = "cuda" if torch.cuda.is_available(
            ) else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device

        self.model_name_or_path = model_name_or_path
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length

        self.model = model
        self.tokenizer = tokenizer
        self.results = {}

        # self.model.to(self.device)

        self.model_info_dict = {
            "module": "TEXT2VEC",
            "model_arch": str(self.model_arch),
            "encoder_type": str(self.encoder_type) if self.encoder_type else None,
        }

    def __str__(self):
        return f"<" \
               f"model_arch: {self.model_arch}, " \
               f"model_name_or_path: {self.model_name_or_path}, " \
               f"max_seq_length: {self.max_seq_length}, " \
               f"encoder_type: {self.encoder_type}, " \
               f"" \
               f">"

    def clear(self):
        # del self.model
        torch.cuda.empty_cache()

    def save_model(self, output_dir, model, results=None):
        logger.info(f"Saving model checkpoint to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        model_info_file_path = os.path.join(output_dir, ".json")
        with open(model_info_file_path, "w", encoding='utf-8') as writer:
            json.dump(self.model_info_dict, writer)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))


class BaseSentenceModel(BaseModel):
    def __init__(
        self,
        model_arch: Union[str, ModelArch] = None,
        model_name_or_path: str = None,
        num_classes: int = 2,
        max_seq_length: int = 128,
        encoder_type: Union[str, EncoderType] = None,
        device: Optional[str] = None,
    ):
        '''
        参数校验
        '''
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)

        super().__init__(
            model_arch,
            model_name_or_path,
            num_classes,
            max_seq_length,
            encoder_type,
            device,
            model,
            tokenizer,
        )

        self.model.to(self.device)
    
    def get_sentence_embedding_dimension(self):
        return getattr(self.model.pooler.dense, "out_features", None)

    def get_sentence_embeddings(self, input_ids, attention_mask, token_type_ids):
        model_output = self.model(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.encoder_type == EncoderType.FIRST_LAST_AVG:
            first = model_output.hidden_states[1]
            last = model_output.hidden_states[-1]
            seq_length = first.size(1)

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            final_encoding = torch.avg_pool1d(
                torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2),
                kernel_size=2).squeeze(-1)
            return final_encoding

        if self.encoder_type == EncoderType.LAST_AVG:
            sequence_output = model_output.last_hidden_state
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        if self.encoder_type == EncoderType.CLS:
            sequence_output = model_output.last_hidden_state
            return sequence_output[:, 0]

        if self.encoder_type == EncoderType.POOLER:
            return model_output.pooler_output

        if self.encoder_type == EncoderType.MEAN:
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            final_encoding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9)
            return final_encoding

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
    ):
        self.model.eval()
        if device is None:
            device = self.device
        if convert_to_tensor:
            convert_to_numpy = False
        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            with torch.no_grad():
                embeddings = self.get_sentence_embeddings(
                    **self.tokenizer(sentences_batch, max_length=self.max_seq_length,
                                     padding=True, truncation=True, return_tensors='pt').to(device)
                )
            embeddings = embeddings.detach()
            if convert_to_numpy:
                embeddings = embeddings.cpu()
            all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings
    
    def evaluate(self, eval_dataset, output_dir: str = None, batch_size: int = 16):
        results = {}

        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        self.model.to(self.device)
        self.model.eval()

        batch_labels = []
        batch_preds = []
        for batch in tqdm(eval_dataloader, disable=False, desc="Running Evaluation"):
            source, target, labels = batch
            labels = labels.to(self.device)
            batch_labels.extend(labels.cpu().numpy())
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(self.device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(self.device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(self.device)

            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(self.device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(self.device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(self.device)

            with torch.no_grad():
                source_embeddings = self.get_sentence_embeddings(source_input_ids, source_attention_mask,
                                                                 source_token_type_ids)
                target_embeddings = self.get_sentence_embeddings(target_input_ids, target_attention_mask,
                                                                 target_token_type_ids)
                preds = torch.cosine_similarity(source_embeddings, target_embeddings)
            batch_preds.extend(preds.cpu().numpy())

        spearman = compute_spearmanr(batch_labels, batch_preds)
        pearson = compute_pearsonr(batch_labels, batch_preds)
        logger.debug(f"labels: {batch_labels[:10]}")
        logger.debug(f"preds:  {batch_preds[:10]}")
        logger.debug(f"pearson: {pearson}, spearman: {spearman}")

        results["eval_spearman"] = spearman
        results["eval_pearson"] = pearson
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "eval_results.txt"), "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

        return results
    
    def eval_model(self, eval_dataset: Dataset, output_dir: str = None, verbose: bool = True, batch_size: int = 16):
        result = self.evaluate(eval_dataset, output_dir, batch_size=batch_size)
        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result
    
    
    