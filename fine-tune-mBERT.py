import transformers

import argparse
import math
import os

import urllib.request
from tqdm import tqdm

from pathlib import Path

import pandas as pd
import numpy as np

import torch



from thai2transformers.datasets import SequenceClassificationDataset




from thai2transformers import preprocess


from args_func import parser
from conf import DATASET_METADATA, TOKENIZER_CLS, PUBLIC_MODEL
from thai2transformers.conf import Task
from init_func import init_public_model_tokenizer_for_seq_cls, init_model_tokenizer_for_seq_cls, init_trainer
from init_func import _process_transformers
from load_func import load_dataset_for_model, load_label_encoder

from functools import partial

from transformers import (
    DataCollatorWithPadding,
    default_data_collator
)

import scipy
if __name__ == "__main__":
    
    args = parser.parse_args("mbert wisesight_sentiment ./output/ ./log/ --batch_size 8".split())
    # Set seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    
    print(f'\n\n[INFO] Dataset: {args.dataset_name}')
    print(f'\n\n[INFO] Huggingface\'s dataset name: {DATASET_METADATA[args.dataset_name]["huggingface_dataset_name"]} ')
    print(f'[INFO] Task: {DATASET_METADATA[args.dataset_name]["task"].value}')
    print(f'\n[INFO] space_token: {args.space_token}')
    print(f'[INFO] prepare_for_tokenization: {args.prepare_for_tokenization}\n')

    dataset = load_dataset_for_model(args)
    label_encoder = load_label_encoder(args, dataset)

    text_input_col_name = DATASET_METADATA[args.dataset_name]['text_input_col_name']

    task = DATASET_METADATA[args.dataset_name]['task']
    model, tokenizer, config = init_public_model_tokenizer_for_seq_cls(args.tokenizer_type_or_public_model_name,
                                                        task=task,
                                                        num_labels=DATASET_METADATA[args.dataset_name]['num_labels']);
    
    print('\n[INFO] Preprocess and tokenizing texts in datasets')
    max_length = args.max_length if args.max_length else config.max_position_embeddings
    print(f'[INFO] max_length = {max_length} \n')

    dataset_split = { split_name: SequenceClassificationDataset.from_dataset(
                        task,
                        tokenizer,
                        dataset[split_name],
                        text_input_col_name,
                        DATASET_METADATA[args.dataset_name]['label_col_name'],
                        max_length=500,
                        space_token=args.space_token,
                        prepare_for_tokenization=args.prepare_for_tokenization,
                        preprocessor=partial(_process_transformers, 
                            pre_rules = [
                            preprocess.fix_html,
                            preprocess.rm_brackets,
                            preprocess.replace_newlines,
                            preprocess.rm_useless_spaces,
                            partial(preprocess.replace_spaces, space_token=args.space_token) if args.space_token != ' ' else lambda x: x,
                            preprocess.replace_rep_after],
                            lowercase=args.lowercase
                        ),
                        label_encoder=label_encoder) for split_name in DATASET_METADATA[args.dataset_name]['split_names']
                    }

    print('[INFO] Done.')

    ######################################

    warmup_steps = math.ceil(len(dataset_split['train']) / args.batch_size * args.warmup_ratio * args.num_train_epochs)

    print(f'\n[INFO] Number of train examples = {len(dataset["train"])}')
    print(f'[INFO] Number of batches per epoch (training set) = {math.ceil(len(dataset_split["train"]) / args.batch_size)}')

    print(f'[INFO] Warmup ratio = {args.warmup_ratio}')
    print(f'[INFO] Warmup steps = {warmup_steps}')
    print(f'[INFO] Learning rate: {args.learning_rate}')
    print(f'[INFO] Logging steps: {args.logging_steps}')
    print(f'[INFO] FP16 training: {args.fp16}\n')

    # if 'validation' in DATASET_METADATA[args.dataset_name]['split_names']:
    print(f'[INFO] Number of validation examples = {len(dataset["validation"])}')
    print(f'[INFO] Number of batches per epoch (validation set) = {math.ceil(len(dataset_split["validation"]))}')

    data_collator = DataCollatorWithPadding(tokenizer,
                                            padding=True,
                                            pad_to_multiple_of=8 if args.fp16 else None)

    trainer, training_args = init_trainer(task=task,
                                model=model,
                                train_dataset=dataset_split['train'],
                                val_dataset=dataset_split['validation'] if 'validation' in DATASET_METADATA[args.dataset_name]['split_names'] else None,
                                warmup_steps=warmup_steps,
                                args=args,
                                data_collator=data_collator)

    print('[INFO] TrainingArguments:')
    print(training_args)
    print('\n')

    print('\nBegin model finetuning.')
    trainer.train()
    print('Done.\n')

    print('[INDO] Begin saving best checkpoint.')
    trainer.save_model(os.path.join(args.output_dir, 'checkpoint-best'))

    print('[INFO] Done.\n')

    print('\nBegin model evaluation on test set.')

    p, label_ids, result = trainer.predict(test_dataset=dataset_split['test'])

    print(f'Evaluation on test set (dataset: {args.dataset_name})')    

    for key, value in result.items():
        print(f'{key} : {value:.4f}')

    pred = scipy.special.softmax(p, axis=1)
    pred = np.argmax(pred, axis=1)

    darr = []
    for idx, d in enumerate(dataset["test"]):
      d["prediction"] = pred[idx]
      darr.append(d)

    df = pd.DataFrame(darr)
    model_name = args.tokenizer_type_or_public_model_name
    df.to_csv(f"{model_name}.csv", index=False)

