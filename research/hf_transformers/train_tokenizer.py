#!/usr/bin/env python3

import argparse
import os
from glob import glob

from tokenizers import ByteLevelBPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_dir',
    default='data/raw_text', type=str,
    required=True, help='The training directory'
)
parser.add_argument(
    '--output_dir',
    default='models/roberta_vi', type=str,
    help='The directory to save tokenizer config'
)
parser.add_argument(
    '--vocab_size',
    default=52000, type=int, required=True,
    help='Size of the base vocabulary (with the added tokens)'
)
parser.add_argument(
    '--min_frequency',
    default=2, type=int
)

args = parser.parse_args()

paths = glob(os.path.join(args.input_dir, '*.txt'))
tokenizer = ByteLevelBPETokenizer()

# Tokenizer training
tokenizer.train(
    files=paths,
    vocab_size=args.vocab_size,
    min_frequency=args.min_frequency,
    special_tokens=[
        '<s>',
        '<pad>',
        '</s>',
        '<unk>',
        '<mask>'
    ]
)

# Saving tokenizer's vocab and config
tokenizer.save(args.output_dir)
