# -*- coding: utf-8 -*-

import os
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from modules.model.Embedding.text2vec.dataset.base import BaseDataset


class BertMatchTrainDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 64):
        super().__init__(tokenizer, data, max_len)

    def text_2_id(self, text_1: str, text_2: str = None):
        return self.tokenizer(text_1, text_2, max_length=self.max_len * 2, truncation=True,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line[0], line[1]), line[2]


class BertMatchTestDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 64):
        super().__init__(tokenizer, data, max_len)

    def text_2_id(self, text_1: str, text_2: str = None):
        return self.tokenizer(text_1, text_2, max_length=self.max_len * 2, truncation=True,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line[0], line[1]), line[2]
