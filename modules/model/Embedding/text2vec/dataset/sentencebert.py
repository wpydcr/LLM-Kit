# -*- coding: utf-8 -*-

import os
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from modules.model.Embedding.text2vec.dataset.base import BaseDataset


class SentenceBertTrainDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 64):
        super().__init__(tokenizer, data, max_len)


class SentenceBertTestDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 64):
        super().__init__(tokenizer, data, max_len)
