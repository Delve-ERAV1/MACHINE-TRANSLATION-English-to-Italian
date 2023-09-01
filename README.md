# Bilingual Translation with Transformers: English to Italian

This project focuses on training a transformer model from scratch to perform bilingual translation between English and Italian. The codebase is implemented using PyTorch-Lightning.

## Table of Contents

- [Objective](#objective)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation and Results](#evaluation-and-results)
- [Metrics Definitions](#metrics-definitions)
- [Sample Predictions](#sample-predictions)
- [Future Work](#future-work)
- [References](#references)

## Objective

- Understand the internal structure of transformers.
- Train a transformer model to achieve a loss of less than 4 over 10 epochs.

## Requirements

- Python 3.7+
- PyTorch
- PyTorch-Lightning

## Setup, Installation, & Training

```bash
pip install pytorch-lightning
python main.py
```

## Data Preparation

```python
def get_ds(config):
    # It only has the train split, we divide ourselves
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=8)
    return(train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt)
```

## Model Architecture

The model is based on the transformer architecture and has the following specifications:
- **Encoder and Decoder Blocks**: 10 blocks each.
- **Attention Heads**: 6 multi-attention heads.
- **Total Parameters**: Approximately 125 million.

```
  | Name    | Type             | Params
---------------------------------------------
0 | model   | Transformer      | 125 M 
1 | loss_fn | CrossEntropyLoss | 0     
---------------------------------------------
125 M     Trainable params
0         Non-trainable params
125 M     Total params
502.022   Total estimated model params size (MB)
```

## Training

- **Batch Size**: 16
- **Optimizer**: Adam
- **Learning Rate**: Starting at 10^-4
Training logs may be found [here](train_logs.txt)

## Evaluation and Results

The model's performance is evaluated using the following metrics:

## Metrics

- **Character Error Rate (CER)**: It measures the performance of character-level recognition, calculating the edit distance between the predicted sequences and the reference sequences.

- **Word Error Rate (WER)**: It is a measure of the performance of a speech recognition or machine translation system. The WER is derived from the Levenshtein distance, working at the word level instead of the phoneme level.

- **BLEU scores**: BLEU (Bilingual Evaluation Understudy) is a metric for evaluating a generated sentence to a reference sentence. A perfect match results in a score of 1.0, while a perfect mismatch results in a score of 0.0.

## Sample Predictions

Throughout the training process, sample predictions are logged every epoch to monitor the model's translation capabilities. These predictions provide insights into how well the model is generalizing and translating unseen data. 

The logs may be found [here](predict_logs.txt)

## Future Work

In the upcoming sessions, we will explore techniques to optimize the training process and potentially speed up the code by up to 5 times.

## References

- https://arxiv.org/abs/1706.03762
- https://lightning.ai/docs/pytorch
- https://www.kaggle.com/code/arunimbasak/transformers-from-scratch/notebook
