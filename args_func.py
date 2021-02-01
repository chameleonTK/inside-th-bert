import argparse

parser = argparse.ArgumentParser()
# Required
parser.add_argument('tokenizer_type_or_public_model_name', type=str, help='The type token model used. Specify the name of tokenizer either `spm`, `newmm`, `syllable`, or `sefr_cut`.')
parser.add_argument('dataset_name', help='Specify the dataset name to finetune. Currently, sequence classification datasets include `wisesight_sentiment`, `generated_reviews_enth-correct_translation`, `generated_reviews_enth-review_star` and`wongnai_reviews`.')
parser.add_argument('output_dir', type=str)
parser.add_argument('log_dir', type=str)

parser.add_argument('--model_dir', type=str)
parser.add_argument('--tokenizer_dir', type=str)
parser.add_argument('--prepare_for_tokenization', action='store_true', default=False, help='To replace space with a special token e.g. `<_>`. This may require for some pretrained models.')
parser.add_argument('--space_token', type=str, default=' ', help='The special token for space, specify if argumet: prepare_for_tokenization is applied')
parser.add_argument('--max_length', type=int, default=None)
parser.add_argument('--lowercase', action='store_true', default=False)

# Finetuning
parser.add_argument('--num_train_epochs', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=1e-05)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--warmup_ratio', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--greater_is_better', action='store_true', default=True)
parser.add_argument('--metric_for_best_model', type=str, default='f1_micro')
parser.add_argument('--logging_steps', type=int, default=10)
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--fp16_opt_level', type=str, default='O1')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--adam_epsilon', type=float, default=1e-08)
parser.add_argument('--max_grad_norm', type=float, default=1.0)

