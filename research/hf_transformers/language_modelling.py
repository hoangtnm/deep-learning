#!/usr/bin/env python3

import argparse
import json
import logging
import os
import pickle
import random
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from apex import amp
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import (DataLoader, Dataset, DistributedSampler,
                              RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, AdamW, PreTrainedTokenizer,
                          get_linear_schedule_with_warmup)

from parameters import HyperParams
from registry import (model_class_lm_dict, model_config_dict,
                      model_tokenizer_dict)

logger = logging.getLogger(__name__)
params = HyperParams()


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - \
            (tokenizer.max_len - tokenizer.max_len_single_sentence)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" +
            str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info(
                f'Loading features from cached file {cached_features_file}'
            )
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(f'Creating features from dataset file at {directory}')

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(text))

            # Truncate in block of block_size
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(tokenizer.build_inputs_with_special_tokens(
                    tokenized_text[i: i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info(
                f'Saving features into cached file {cached_features_file}'
            )
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True,
        help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="The output directory where \
            the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--model_type", type=str, required=True,
        help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file", default=None, type=str,
        help="An optional input evaluation data file\
            to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line", action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    # parser.add_argument(
    #     "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    # )
    parser.add_argument(
        "--model_name_or_path", default=None, type=str,
        help="The model checkpoint for weights initialization.\
            Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true",
        help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name", default=None, type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name", default=None, type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )

    parser.add_argument(
        "--block_size", default=-1, type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true",
        help="Run evaluation during training at each logging step."
    )

    parser.add_argument(
        "--overwrite_cache", action="store_true",
        help="Overwrite the cached training and evaluation sets"
    )
