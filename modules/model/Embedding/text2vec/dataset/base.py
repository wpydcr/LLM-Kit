# -*- coding: utf-8 -*-

import os
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import json


class BaseDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 64):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len, truncation=True,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line[0]), self.text_2_id(line[1]), line[2]

    @staticmethod
    def load_train_data(path):
        data = []
        if not os.path.isfile(path):
            return data
        if path.endswith('.json'):
            with open(path,'r',encoding='utf8') as f:
                line = json.loads(f.readline())
                data.append((line['output']['sentence1'],line['output']['sentence2'],int(line['output']['label'])))
        else:
            with open(path, 'r', encoding='utf8') as f:
                for line in f:
                    line = line.strip().split('\t')
                    if len(line) != 3:
                        logger.warning(f'line size not match, pass: {line}')
                        continue
                    score = int(line[2])
                    if 'STS' in path.upper():
                        score = int(score > 2.5)
                    data.append((line[0], line[1], score))
        return data

    @staticmethod
    def load_test_data(path):
        data = []
        if not os.path.isfile(path):
            return data
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().split('\t')
                if len(line) != 3:
                    logger.warning(f'line size not match, pass: {line}')
                    continue
                score = int(line[2])
                if 'STS' in path.upper():
                    score = int(score > 2.5)
                data.append((line[0], line[1], score))
        return data
