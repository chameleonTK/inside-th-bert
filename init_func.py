from conf import PUBLIC_MODEL, METRICS, DATASET_METADATA
from thai2transformers.conf import Task
from functools import partial
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup, 
    get_constant_schedule, 
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification, 
    AutoConfig,
    Trainer, 
    TrainingArguments,
    CamembertTokenizer,
    BertTokenizer,
    BertTokenizerFast,
    BertConfig,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
    XLMRobertaConfig,
    DataCollatorWithPadding,
    default_data_collator
)

from thai2transformers import preprocess
from typing import Collection, Callable
from thai2transformers.auto import AutoModelForMultiLabelSequenceClassification

def init_public_model_tokenizer_for_seq_cls(public_model_name, task, num_labels):
    config = PUBLIC_MODEL[public_model_name]['config']
    config.num_labels = num_labels
    tokenizer = PUBLIC_MODEL[public_model_name]['tokenizer']
    model_name = PUBLIC_MODEL[public_model_name]['name']
    if task == Task.MULTICLASS_CLS:
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                   config=config)
    if task == Task.MULTILABEL_CLS:
        model = AutoModelForMultiLabelSequenceClassification.from_pretrained(model_name,
                                                                             config=config)

    print(f'\n[INFO] Model architecture: {model} \n\n')
    print(f'\n[INFO] tokenizer: {tokenizer} \n\n')

    return model, tokenizer, config

def init_model_tokenizer_for_seq_cls(model_dir, tokenizer_cls, tokenizer_dir, task, num_labels):
    
    config = AutoConfig.from_pretrained(
        model_dir,
        num_labels=num_labels
    );

    tokenizer = tokenizer_cls.from_pretrained(
        tokenizer_dir,
    );

    if task == Task.MULTICLASS_CLS:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            config=config,
        )
    elif task == Task.MULTILABEL_CLS:
        model = AutoModelForMultiLabelSequenceClassification.from_pretrained(
            model_dir,
            config=config,
        )

    # print(f'\n[INFO] Model architecture: {model} \n\n')
    # print(f'\n[INFO] tokenizer: {tokenizer} \n\n')

    return model, tokenizer, config

def init_trainer(task, model, train_dataset, val_dataset, warmup_steps, args, data_collator=default_data_collator): 
        
    training_args = TrainingArguments(
                        num_train_epochs=args.num_train_epochs,
                        per_device_train_batch_size=args.batch_size,
                        per_device_eval_batch_size=args.batch_size,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        learning_rate=args.learning_rate,
                        warmup_steps=warmup_steps,
                        weight_decay=args.weight_decay,
                        adam_epsilon=args.adam_epsilon,
                        max_grad_norm=args.max_grad_norm,
                        #checkpoint
                        output_dir=args.output_dir,
                        overwrite_output_dir=True,
                        #logs
                        logging_dir=args.log_dir,
                        logging_first_step=False,
                        logging_steps=args.logging_steps,
                        #eval
                        evaluation_strategy='epoch' if 'validation' in DATASET_METADATA[args.dataset_name]['split_names'] else 'no',
                        load_best_model_at_end=True,
                        #others
                        seed=args.seed,
                        fp16=args.fp16,
                        fp16_opt_level=args.fp16_opt_level,
                        dataloader_drop_last=False,
                        no_cuda=args.no_cuda,
                        metric_for_best_model=args.metric_for_best_model,
                        prediction_loss_only=False,
                        run_name="X"
                    )
    if task == Task.MULTICLASS_CLS:
        compute_metrics_fn = METRICS[task]
    elif task == Task.MULTILABEL_CLS:
        compute_metrics_fn = partial(METRICS[task],n_labels=DATASET_METADATA[args.dataset_name]['num_labels'])

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    return trainer, training_args

def _process_transformers(
    text: str,
    pre_rules: Collection[Callable] = [
        preprocess.fix_html,
        preprocess.rm_brackets,
        preprocess.replace_newlines,
        preprocess.rm_useless_spaces,
        preprocess.replace_spaces,
        preprocess.replace_rep_after,
    ],
    tok_func: Callable = preprocess.word_tokenize,
    post_rules: Collection[Callable] = [preprocess.ungroup_emoji, preprocess.replace_wrep_post],
    lowercase: bool = False
) -> str:
    if lowercase:
        text = text.lower()
    for rule in pre_rules:
        text = rule(text)
    toks = tok_func(text)
    for rule in post_rules:
        toks = rule(toks)
    return "".join(toks)
