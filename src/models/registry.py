"""
Model Registry for Multilingual Marked Content Text Joining
Maps target languages to their specialized language-specific models (Language-Specific Strategy),
defines the universal multilingual model architecture (Universal Strategy),
and supports hybrid cross-encoder routing (Mixed Strategy).
"""

from typing import Dict, Any

# Primary specialized pretrained transformer per target language (Language-Specific Strategy)
LANGUAGE_SPECIFIC_ENCODERS: Dict[str, Dict[str, Any]] = {
    "pt": {
        "name": "Portuguese BERTimbau",
        "hf_model": "neuralmind/bert-base-portuguese-cased",
        "description": "Monolingual Brazilian Portuguese BERT base cased"
    },
    "en": {
        "name": "English MiniLM-L6",
        "hf_model": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Ultra-lightweight English transformer (22M params, fast sequence classification)"
    },
    "ru": {
        "name": "Russian RuBERT-Tiny2",
        "hf_model": "cointegrated/rubert-tiny2",
        "description": "Ultra-compact Russian transformer (29M params, 3x faster than base)"
    },
    "es": {
        "name": "Spanish BETO",
        "hf_model": "dccuchile/bert-base-spanish-wwm-cased",
        "description": "Monolingual Spanish BERT base cased"
    },
    "fr": {
        "name": "French CamemBERT",
        "hf_model": "almanach/camembert-base",
        "description": "Monolingual French RoBERTa base"
    },
    "de": {
        "name": "German BERT",
        "hf_model": "dbmdz/bert-base-german-cased",
        "description": "Monolingual German BERT base cased"
    },
    "cs": {
        "name": "Czech RobeCzech",
        "hf_model": "ufal/robeczech-base",
        "description": "Monolingual Czech RoBERTa base cased (Hybrid Book native)"
    },
    "tr": {
        "name": "Turkish BERT",
        "hf_model": "dbmdz/bert-base-turkish-cased",
        "description": "Monolingual Turkish BERT base cased"
    },
    "pl": {
        "name": "Polish HerBERT",
        "hf_model": "allegro/herbert-base-cased",
        "description": "Monolingual Polish RoBERTa base cased"
    },
    "uk": {
        "name": "Ukrainian RoBERTa",
        "hf_model": "youscan/ukr-roberta-base",
        "description": "Monolingual Ukrainian RoBERTa base cased"
    },
    "id": {
        "name": "Indonesian IndoBERT",
        "hf_model": "indobenchmark/indobert-base-p1",
        "description": "Monolingual Indonesian BERT base cased"
    }
}

# Universal multilingual transformer candidate (Universal Strategy)
UNIVERSAL_MULTILINGUAL_ENCODER = {
    "name": "Multilingual MiniLM-L12",
    "hf_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "description": "Multilingual 50+ language transformer (117M params, INT8 browser optimized)"
}
