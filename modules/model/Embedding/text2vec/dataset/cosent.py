# -*- coding: utf-8 -*-

import os
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from modules.model.Embedding.text2vec.dataset.base import BaseDataset


class CosentTrainDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 64):
        super().__init__(tokenizer, data, max_len)

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line[0]), line[1]

    @staticmethod
    def load_train_data(path):
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
                data.append((line[0], score))
                data.append((line[1], score))
        return data


class CosentTestDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 64):
        super().__init__(tokenizer, data, max_len)
