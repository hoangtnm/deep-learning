#!/usr/bin/env python3

import argparse
import copy
import json
import logging
import os
import pickle
import random
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from apex import amp
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (DataLoader, Dataset, DistributedSampler,
                              RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, AdamW, PreTrainedModel,
                          PreTrainedTokenizer, get_linear_schedule_with_warmup)

from parameters import HyperParams
from registry import (model_class_lm_dict, model_config_dict,
                      model_tokenizer_dict)

logger = logging.getLogger(__name__)
params = HyperParams()


def create_configs(args):
    config = {
        "architectures": [
            "RobertaForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 514,
        "model_type": "roberta",
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "vocab_size": 52000
    }

    tokenizer_config = {
        "max_len": 512
    }

    with open(os.path.join(args.input_dir, 'config.json'), 'w') as f:
        json.dump(config, f)
    with open(os.path.join(args.input_dir, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer_config, f)


# DEPREACATED
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


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args,
                 file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        logger.info(f'Creating features from dataset file at {file_path}')

        with open(file_path, encoding='utf-8') as f:
            lines = [line for line in f.read().splitlines()
                     if (len(line) > 0 and not line.isspace())]

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info(
                f'Loading features from cached file {cached_features_file}'
            )
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            self.examples = tokenizer.batch_encode_plus(
                lines, add_special_tokens=True, max_length=block_size
            )['input_ids']

            logger.info(
                f'Saving features into cached file {cached_features_file}'
            )
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path, args.block_size)
    else:
        return TextDataset(tokenizer, args, file_path, args.block_size)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepares masked tokens inputs/labels for masked language modeling:
    80% MASK, 10% random, 10% original."""

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary \
                for masked language modeling. \
                Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(
        special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(
        labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model: PreTrainedModel,
          tokenizer: PreTrainedTokenizer):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    writer = SummaryWriter()
    args.n_gpu = params.n_gpu
    args.per_gpu_train_batch_size = params.per_gpu_train_batch_size
    args.train_batch_size = args.per_gpu_train_batch_size * \
        max(1, params.n_gpu)
    args.gradient_accumulation_steps = params.gradient_accumulation_steps
    args.max_grad_norm = params.max_grad_norm
    if not args.num_train_epochs:
        args.num_train_epochs = params.num_train_epochs
    args.learning_rate = params.learning_rate
    args.weight_decay = params.weight_decay
    args.adam_epsilon = params.adam_epsilon
    args.warmup_steps = params.warmup_steps
    args.fp16 = params.fp16

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True,
                            padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, args.train_batch_size,
                                  sampler=train_sampler, collate_fn=collate,
                                  num_workers=4)

    t_total = len(train_dataloader) \
        // args.gradient_accumulation_steps \
        * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=params.fp16_opt_level)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # TODO: Loading checkpoint for AMP
    # Train!
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(
        f'  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}'
    )
    logger.info(
        '  Total train batch size (w. parallel, & accumulation) = %d',
        args.train_batch_size * args.gradient_accumulation_steps
    )
    logger.info(
        f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}'
    )
    logger.info(
        f'  Total optimization steps = {t_total}'
    )

    global_step = 0
    best_perplexity = 0.0
    training_loss, running_loss = 0.0, 0.0

    # Take care of distributed/parallel training
    model_to_resize = model.module if hasattr(model, "module") else model
    model_to_resize.resize_token_embeddings(len(tokenizer))
    model.train()

    for epoch in range(args.num_train_epochs):
        print(f'Epoch {epoch}/{args.num_train_epochs - 1}')
        print('-' * 10)

        for step, batch in enumerate(tqdm(train_dataloader)):
            inputs, labels = mask_tokens(batch, tokenizer, args) \
                if args.mlm else (batch, batch)
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs, masked_lm_labels=labels) \
                if args.mlm else model(inputs, labels=labels)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            training_loss += loss.item()
            running_loss += loss.item() * inputs.size(0)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                             args.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(),
                                             args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                global_step += 1

                # TODO: args.evaluate_during_training
                writer.add_scalar('learning_rate',
                                  scheduler.get_lr()[0], global_step)
                writer.add_scalar('loss/training',
                                  training_loss, global_step)
                training_loss = 0.0

        epoch_loss = running_loss / len(train_dataset)
        # TODO: Evaluates and saves checkpoint after every epoch
        result = evaluate(args, model, tokenizer)
        epoch_perplexity = result.get('perplexity')

        if step == 0:
            best_perplexity = epoch_perplexity
        else:
            if epoch_perplexity < best_perplexity:
                best_perplexity = epoch_perplexity

        print(f'Loss: {epoch_loss:.4f} perplexity:{epoch_perplexity}')

    writer.close()

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print(f'Perplexity: {best_perplexity}')

    return model


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
             prefix='') -> Dict:

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    args.eval_batch_size = params.per_gpu_val_batch_size * \
        max(1, params.eval_batch_size)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True,
                            padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, args.eval_batch_size,
        sampler=eval_sampler, collate_fn=collate, num_workers=4
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info(f'  Num examples = {len(eval_dataset)}')
    logger.info(f'  Batch size = {args.eval_batch_size}')
    eval_loss = 0.0
    model.eval()

    for step, batch in enumerate(tqdm(eval_dataloader, desc='Evaluating')):
        inputs, labels = mask_tokens(batch, tokenizer, args) \
            if args.mlm else (batch, batch)
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) \
                if args.mlm else model(inputs, labels=labels)
            loss = outputs[0]
            eval_loss += loss.mean().item()

    eval_loss = eval_loss / (step + 1)
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}
    return result


def main():
    parser = argparse.ArgumentParser()

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

    parser.add_argument(
        "--num_train_epochs", type=int,
        help="Total number of training epochs to perform."
    )

    args = parser.parse_args()
    args.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    # create_configs(args)
    config_class = model_config_dict[args.model_type]
    model_class = model_class_lm_dict[args.model_type]
    tokenizer_class = model_tokenizer_dict[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported"
            "and load it from here, using --tokenizer_name".format(
                tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        # Our input block size will be the max possible for the model
        args.block_size = tokenizer.max_len
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            config=config)
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)
    model.to(args.device)

    logger.info(f'Training/evaluation parameters {args}')

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer)
        model = train(args, train_dataset, model, tokenizer)

    # Saving best checkpoint
    if args.do_train:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f'Saving model checkpoint to {args.output_dir}')
        # Save a trained model, configuration and tokenizer.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        # Good practice: save your training arguments
        # together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    if args.do_eval:
        result = evaluate(args, model, tokenizer, prefix='')

    return result


if __name__ == '__main__':
    main()
