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

from modules.model.Embedding.text2vec.model.base import BaseSentenceModel
from modules.model.Embedding.text2vec.utils.base import compute_pearsonr, compute_spearmanr, set_seed, ModelArch, EncoderType


class SentenceBertModel(BaseSentenceModel):
    def __init__(
        self,
        model_name_or_path: str = None,
        num_classes: int = 2,
        max_seq_length: int = 128,
        encoder_type: str = "MEAN",
        device: str = None,
    ):
        super().__init__(
            model_arch = "SENTENCEBERT",
            model_name_or_path = model_name_or_path,
            max_seq_length = max_seq_length,
            encoder_type = encoder_type,
            device = device,
        )
        self.is_stop = False
        self.classifier = nn.Linear(self.model.config.hidden_size * 3, num_classes).to(self.device)

    def concat_embeddings(self, source_embeddings, target_embeddings):
        # (u, v, |u - v|)
        embs = [source_embeddings, target_embeddings, torch.abs(source_embeddings - target_embeddings)]
        input_embs = torch.cat(embs, 1)
        # fc layer
        logits = self.classifier(input_embs)
        return logits

    def calc_loss(self, y_true, y_pred):
        loss = nn.CrossEntropyLoss()(y_pred, y_true)
        return loss

    def train(
        self,
        train_dataset: Dataset,
        output_dir: str,
        eval_dataset: Dataset = None,
        verbose: bool = True,
        batch_size: int = 8,
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
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Use pytorch device: {}".format(self.device))
        self.model.to(self.device)
        set_seed(seed)

        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
        total_steps = len(train_dataloader) * num_epochs
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        warmup_steps = math.ceil(total_steps * warmup_ratio)  # by default 10% of _train data for warm-up
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Num steps = {total_steps}")
        logger.info(f"  Warmup-steps: {warmup_steps}")

        logger.info("  Training started")
        global_step = 0
        self.model.zero_grad()
        epoch_number = 0
        best_eval_metric = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if self.model_name_or_path and os.path.exists(self.model_name_or_path):
            try:
                # set global_step to global_step of last saved checkpoint from model path
                checkpoint_suffix = self.model_name_or_path.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (len(train_dataloader) // gradient_accumulation_steps)
                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d" % epochs_trained)
                logger.info("   Continuing training from global step %d" % global_step)
                logger.info("   Will skip the first %d steps in the current epoch" % steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        training_progress_scores = {
            "global_step": [],
            "train_loss": [],
            "eval_spearman": [],
            "eval_pearson": [],
        }
        for current_epoch in trange(int(num_epochs), desc="Epoch", disable=False, mininterval=0):
            self.model.train()
            current_loss = 0
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            batch_iterator = tqdm(train_dataloader,
                                  desc=f"Running Epoch {epoch_number + 1} of {num_epochs}",
                                  disable=False,
                                  mininterval=0)
            for step, batch in enumerate(batch_iterator):
                if self.is_stop:
                    break
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                source, target, labels = batch
                # source        [batch, 1, seq_len] -> [batch, seq_len]
                source_input_ids = source.get('input_ids').squeeze(1).to(self.device)
                source_attention_mask = source.get('attention_mask').squeeze(1).to(self.device)
                source_token_type_ids = source.get('token_type_ids').squeeze(1).to(self.device)
                # target        [batch, 1, seq_len] -> [batch, seq_len]
                target_input_ids = target.get('input_ids').squeeze(1).to(self.device)
                target_attention_mask = target.get('attention_mask').squeeze(1).to(self.device)
                target_token_type_ids = target.get('token_type_ids').squeeze(1).to(self.device)
                labels = labels.to(self.device)

                # get sentence embeddings of BERT encoder
                source_embeddings = self.get_sentence_embeddings(source_input_ids, source_attention_mask,
                                                                 source_token_type_ids)
                target_embeddings = self.get_sentence_embeddings(target_input_ids, target_attention_mask,
                                                                 target_token_type_ids)
                logits = self.concat_embeddings(source_embeddings, target_embeddings)
                loss = self.calc_loss(labels, logits)
                current_loss = loss.item()
                if verbose:
                    batch_iterator.set_description(
                        f"Epoch: {epoch_number + 1}/{num_epochs}, Batch:{step}/{len(train_dataloader)}, Loss: {current_loss:9.4f}")

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1
            if self.is_stop:
                break
            epoch_number += 1
            if (current_epoch + 1) % logging_epochs <= 0:
                output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))
                results = {}
                if eval_dataset:
                    results = self.eval_model(eval_dataset, output_dir_current, verbose=verbose, batch_size=batch_size)
                self.save_model(output_dir_current, model=self.model, results=results)
                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                if eval_dataset and results:
                    for key in results:
                        training_progress_scores[key].append(results[key])
                else:
                    training_progress_scores["eval_spearman"].append(np.nan)
                    training_progress_scores["eval_pearson"].append(np.nan)
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(output_dir, "training_progress_scores.csv"), index=False)

                best_eval_metric = self.save_best_or_last_model(eval_dataset,results,final_model,best_eval_metric,current_loss)
                yield global_step, training_progress_scores
                
            if 0 < max_steps < global_step:
                break
        self.save_best_or_last_model(eval_dataset,results,final_model,best_eval_metric,current_loss,save_last=True)
        yield global_step, training_progress_scores

    def save_best_or_last_model(self, eval_dataset,results,final_model,best_eval_metric,current_loss,save_last=False):
        if eval_dataset and results:
            eval_spearman = results["eval_spearman"]
            if eval_spearman > best_eval_metric:
                best_eval_metric = eval_spearman
                logger.info(f"Save new best model, best_eval_metric: {best_eval_metric}")
                self.save_model(f'models/Embedding/{final_model}', model=self.model, results=results)
        elif save_last:
            logger.info(f"Save new model, current_loss: {current_loss}")
            self.save_model(f'models/Embedding/{final_model}', model=self.model, results=None)

        return best_eval_metric
