from datasets import load_dataset, list_metrics, Dataset
from conf import DATASET_METADATA
from thai2transformers.conf import Task
from sklearn import preprocessing
from thai2transformers.utils import get_dict_val

def load_dataset_for_model(args):
    if args.dataset_name == 'wongnai_reviews':
        print(f'\n\n[INFO] For Wongnai reviews dataset, perform train-val set splitting (0.9,0.1)')
        dataset = load_dataset(DATASET_METADATA[args.dataset_name]["huggingface_dataset_name"])
        print(f'\n\n[INFO] Perform dataset splitting')
        train_val_split = dataset['train'].train_test_split(test_size=0.1, shuffle=True, seed=args.seed)
        dataset['train'] = train_val_split['train']
        dataset['validation'] = train_val_split['test']
        print(f'\n\n[INFO] Done')
        print(f'dataset: {dataset}')
    else:
        dataset = load_dataset(DATASET_METADATA[args.dataset_name]["huggingface_dataset_name"])
    
    return dataset


def load_label_encoder(args, dataset):
    if DATASET_METADATA[args.dataset_name]['task'] == Task.MULTICLASS_CLS:
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(get_dict_val(dataset['train'], keys=DATASET_METADATA[args.dataset_name]['label_col_name']))
    else:
        label_encoder = None
    
    return label_encoder
