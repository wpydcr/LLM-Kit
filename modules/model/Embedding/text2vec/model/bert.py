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

from modules.model.Embedding.text2vec.model.base import BaseModel
from modules.model.Embedding.text2vec.utils.base import compute_pearsonr, compute_spearmanr, set_seed, ModelArch, EncoderType


class BertMatchModel(BaseModel):
    class BertMatchModule(nn.Module):
        def __init__(
            self,
            model_name_or_path: str = "bert-base-chinese",
            num_classes: int = 2,
            device: Optional[str] = None,
        ):
            super().__init__()
            self.bert = BertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_classes)

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            self.device = device
            
            self.bert.to(self.device)

        def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
            outputs = self.bert(input_ids, token_type_ids, attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            probs = nn.functional.softmax(logits, dim=-1)
            return loss, logits, probs


    def __init__(
        self,
        model_name_or_path: str = None,
        num_classes: int = 2,
        max_seq_length: int = 128,
        encoder_type = None,
        device: Optional[str] = None,
    ):
        model = self.BertMatchModule(model_name_or_path, num_classes, device)
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.is_stop = False
        super().__init__(
            "BERT",
            model_name_or_path,
            num_classes,
            max_seq_length,
            encoder_type,
            device,
            model,
            tokenizer,
        )

    def evaluate(self, eval_dataset, output_dir: str = None, batch_size: int = 16):
        results = {}

        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        self.model.bert.to(self.device)
        self.model.bert.eval()

        batch_labels = []
        batch_preds = []
        for batch in tqdm(eval_dataloader, disable=False, desc="Running Evaluation"):
            inputs, labels = batch
            labels = labels.to(self.device)
            batch_labels.extend(labels.cpu().numpy())
            # inputs        [batch, 1, seq_len] -> [batch, seq_len]
            input_ids = inputs.get('input_ids').squeeze(1).to(self.device)
            attention_mask = inputs.get('attention_mask').squeeze(1).to(self.device)
            token_type_ids = inputs.get('token_type_ids').squeeze(1).to(self.device)

            with torch.no_grad():
                loss, logits, probs = self.model(input_ids, attention_mask, token_type_ids, labels)
            batch_preds.extend(probs.cpu().numpy())
        
        batch_preds = list(map(lambda x:x.max(), batch_preds))
        
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
        final_model:str = None
    ):
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Use pytorch device: {}".format(self.device))
        self.model.bert.to(self.device)
        set_seed(seed)

        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
        total_steps = len(train_dataloader) * num_epochs
        param_optimizer = list(self.model.bert.named_parameters())
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
        self.model.bert.zero_grad()
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
            self.model.bert.train()
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
                inputs, labels = batch
                labels = labels.to(self.device)
                # inputs        [batch, 1, seq_len] -> [batch, seq_len]
                input_ids = inputs.get('input_ids').squeeze(1).to(self.device)
                attention_mask = inputs.get('attention_mask').squeeze(1).to(self.device)
                token_type_ids = inputs.get('token_type_ids').squeeze(1).to(self.device)
                loss, logits, probs = self.model(input_ids, attention_mask, token_type_ids, labels)
                current_loss = loss.item()

                if verbose:
                    batch_iterator.set_description(
                        f"Epoch: {epoch_number + 1}/{num_epochs}, Batch:{step}/{len(train_dataloader)}, Loss: {current_loss:9.4f}")

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.bert.parameters(), max_grad_norm)
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
                self.save_model(output_dir_current, model=self.model.bert, results=results)
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
                self.save_model(f'models/Embedding/{final_model}', model=self.model.bert, results=results)
        elif save_last:
            logger.info(f"Save new model, current_loss: {current_loss}")
            self.save_model(f'models/Embedding/{final_model}', model=self.model.bert, results=None)

        return best_eval_metric
