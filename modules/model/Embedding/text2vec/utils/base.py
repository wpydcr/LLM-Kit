# -*- coding: utf-8 -*-

from enum import Enum
from loguru import logger
import random
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch


def set_seed(seed):
    """
    Set seed for random number generators.
    """
    logger.info(f"Set seed for random, numpy and torch: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize(vecs):
    """
    L2标准化
    """
    norms = (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_spearmanr(x, y):
    """
    Spearman相关系数
    """
    return spearmanr(x, y).correlation


def compute_pearsonr(x, y):
    """
    Pearson系数
    """
    return pearsonr(x, y)[0]


class EncoderType(Enum):
    FIRST_LAST_AVG = 0
    LAST_AVG = 1
    CLS = 2
    POOLER = 3
    MEAN = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EncoderType[s]
        except KeyError:
            raise ValueError()


class ModelArch(Enum):
    COSENT = 0
    SENTENCEBERT = 1
    BERT = 2

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return ModelArch[s]
        except KeyError:
            raise ValueError()


if __name__ == "__main__":
    s = 'FIRST_LAST_AVG'
    # s = None
    e = EncoderType.from_string(s) if isinstance(s, str) else s

    if e not in list(EncoderType) and e is not None:
        raise ValueError(f"{list(EncoderType)} or {None}")

    d = {"encoder_type": str(e)}

    print(d)

    import json
    print(json.dumps(d))
