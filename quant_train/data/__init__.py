"""数据加载模块 —— 加载 HuggingFace 数据集并进行 tokenize"""
from typing import Optional

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def load_and_tokenize(
    dataset_name: str,
    subset: Optional[str] = None,
    text_column: str = "text",
    tokenizer_name: Optional[str] = None,
    max_seq_length: int = 512,
    val_split: float = 0.05,
    seed: int = 42,
):
    """加载数据集 → tokenize → 返回 train/val splits。

    在 CPU 上用小样本也能跑通，验证数据管线没问题。
    """
    if tokenizer_name is None:
        tokenizer_name = dataset_name  # 很多 dataset 自带 tokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, subset, split="train")

    def tokenize_fn(examples):
        texts = examples[text_column]
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    # 按行数简单切分（不用 kfolds，因为量化训练对 val 分布不敏感）
    splits = ds.train_test_split(test_size=val_split, seed=seed)
    return splits["train"], splits["val"], tokenizer
