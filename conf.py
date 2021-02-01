from thai2transformers.metrics import classification_metrics, multilabel_classification_metrics
from thai2transformers.conf import Task
from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer,
)

from transformers import (
    CamembertTokenizer,
    BertTokenizer,
    BertTokenizerFast,
    BertConfig,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
    XLMRobertaConfig
)

METRICS = {
    Task.MULTICLASS_CLS: classification_metrics,
    Task.MULTILABEL_CLS: multilabel_classification_metrics
}

PUBLIC_MODEL = {
    'mbert': {
        'name': 'bert-base-multilingual-cased',
        'tokenizer': BertTokenizerFast.from_pretrained('bert-base-multilingual-cased'),
        'config': BertConfig.from_pretrained('bert-base-multilingual-cased'),
    },
    'xlmr': {
        'name': 'xlm-roberta-base',
        'tokenizer': XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base'),
        'config': XLMRobertaConfig.from_pretrained('xlm-roberta-base'),
    },
    'xlmr-large': {
        'name': 'xlm-roberta-large',
        'tokenizer': XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-large'),
        'config': XLMRobertaConfig.from_pretrained('xlm-roberta-base'),
    },
}

TOKENIZER_CLS = {
    'spm_camembert': CamembertTokenizer,
    'spm': ThaiRobertaTokenizer,
    'newmm': ThaiWordsNewmmTokenizer,
    'syllable': ThaiWordsSyllableTokenizer,
    'sefr_cut': FakeSefrCutTokenizer,
}

DATASET_METADATA = {
    'wisesight_sentiment': {
        'huggingface_dataset_name': 'wisesight_sentiment',
        'task': Task.MULTICLASS_CLS,
        'text_input_col_name': 'texts',
        'label_col_name': 'category',
        'num_labels': 4,
        'split_names': ['train', 'validation', 'test']
    },
    'wongnai_reviews': {
        'huggingface_dataset_name': 'wongnai_reviews',
        'task': Task.MULTICLASS_CLS,
        'text_input_col_name': 'review_body',
        'label_col_name': 'star_rating',
        'num_labels': 5,
        'split_names': ['train', 'validation', 'test']
    },
    'prachathai67k': {
        'huggingface_dataset_name': 'prachathai67k',
        # 'url': 'https://archive.org/download/prachathai67k/data.zip',
        'task': Task.MULTILABEL_CLS,
        'text_input_col_name': 'title',
        'label_col_name': ['politics', 'human_rights', 'quality_of_life',
                           'international', 'social', 'environment',
                           'economics', 'culture', 'labor',
                           'national_security', 'ict', 'education'],
        'num_labels': 12,
        'split_names': ['train', 'validation', 'test']
    }
}