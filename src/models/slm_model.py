"""
Tier 3: Sub-1B Small Language Model (SLM) Classifier
Loads tiny decoder-only SLMs (SmolLM2-135M/360M or Qwen2.5-0.5B)
with a sequence classification head for non-autoregressive binary decision making.
"""

import os
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.models.cross_encoder import BoundaryPairDataset


class SLMBoundaryClassifier:
    """Sub-1B SLM Classifier (e.g. SmolLM2-135M or Qwen2.5-0.5B)."""
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-135M",
        device: Optional[str] = None,
        max_length: int = 128
    ):
        self.model_name = model_name
        self.max_length = max_length
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading SLM: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        config = AutoConfig.from_pretrained(model_name, num_labels=2)
        config.pad_token_id = self.tokenizer.pad_token_id
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        ).to(self.device)
        
    def predict_proba(
        self,
        boundaries: List[Dict[str, Any]],
        batch_size: int = 16,
        include_layout_cues: bool = True
    ) -> np.ndarray:
        """Computes probabilities using single-pass classification head."""
        dataset = BoundaryPairDataset(boundaries, self.tokenizer, self.max_length, include_layout_cues)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
                
        return np.vstack(all_probs)
        
    def evaluate(
        self,
        boundaries: List[Dict[str, Any]],
        batch_size: int = 16,
        include_layout_cues: bool = True
    ) -> Dict[str, Any]:
        """Calculates evaluation metrics."""
        probs = self.predict_proba(boundaries, batch_size=batch_size, include_layout_cues=include_layout_cues)
        preds = np.argmax(probs, axis=1)
        y = np.array([int(b['label']) for b in boundaries])
        
        acc = accuracy_score(y, preds)
        macro_f1 = f1_score(y, preds, average='macro')
        join_f1 = f1_score(y, preds, pos_label=1)
        split_f1 = f1_score(y, preds, pos_label=0)
        roc_auc = roc_auc_score(y, probs[:, 1])
        
        return {
            "model_name": self.model_name,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "join_f1": float(join_f1),
            "split_f1": float(split_f1),
            "roc_auc": float(roc_auc)
        }
