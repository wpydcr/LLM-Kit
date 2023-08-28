# -*- coding: utf-8 -*-
"""
本模块为了集成多个embedding模型，使用 子模块.子模型 结构以便于后续扩展
例如：
    |-base
    |-text2vec
        |-bert
        |-sentencebert
        |-cosent
    |-other
        |-other

本模块通过 base 对外暴露 load_model，train_model， eval_model 等方法
可以统一载入、训练、评估 不同子模块不同子模型
其中，
load_model 通过 模型信息文件（.json）载入不同子模块不同子模型
train_model 训练、保存模型，且包含模型信息文件（.json）
eval_model 评估模型

所以本模块所支持的基础模型，需要包含模型信息文件（.json）
模型信息文件对应 ModelInfo 类，包含子模块名称，以及子模型属性
模型信息文件示例：
    text2vec.bert 基础模型 bert-base-uncased
        {"module": "TEXT2VEC", "model_arch": "BERT", "encoder_type": null}
        
    text2vec.sentencebert 基础模型 paraphrase-multilingual-MiniLM-L12-v2
        {"module": "TEXT2VEC", "model_arch": "SENTENCEBERT", "encoder_type": "MEAN"}
        
    text2vec.cosent 基础模型 text2vec-base-chinese
        {"module": "TEXT2VEC", "model_arch": "COSENT", "encoder_type": "FIRST_LAST_AVG"}

"""