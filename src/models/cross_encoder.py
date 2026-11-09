"""
Tier 1: Cross-Encoder Sequence Classifier
Fine-tunes a lightweight encoder (MiniLM, BERTimbau, DeBERTa, or ModernBERT)
to predict binary boundary decisions P(JOIN | Piece A, Piece B).
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


class BoundaryPairDataset(Dataset):
    """PyTorch Dataset for text piece boundary pairs."""
    
    def __init__(
        self,
        boundaries: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 128,
        include_layout_cues: bool = True
    ):
        self.samples = []
        for b in boundaries:
            txt_a = str(b.get('text_a', '')).strip()
            txt_b = str(b.get('text_b', '')).strip()
            label = int(b.get('label', 0))
            
            if include_layout_cues:
                dy = round(float(b.get('bbox_b', [0, 0, 0, 0])[1]) - float(b.get('bbox_a', [0, 0, 0, 0])[3]), 1)
                dx = round(float(b.get('bbox_b', [0, 0, 0, 0])[0]) - float(b.get('bbox_a', [0, 0, 0, 0])[0]), 1)
                same_font = b.get('font_a') == b.get('font_b')
                prefix = f"[dy:{dy} dx:{dx} font:{int(same_font)}] "
                txt_a = prefix + txt_a
                
            self.samples.append((txt_a, txt_b, label))
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        txt_a, txt_b, label = self.samples[idx]
        encoding = self.tokenizer(
            txt_a,
            txt_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item


class CrossEncoderClassifier:
    """Wrapper for training, evaluating, and exporting transformer cross-encoders."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        max_length: int = 128
    ):
        self.model_name = model_name
        self.max_length = max_length
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading tokenizer and model: {model_name} on {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            except Exception:
                from transformers import BertTokenizer
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
    def fit(
        self,
        train_boundaries: List[Dict[str, Any]],
        val_boundaries: Optional[List[Dict[str, Any]]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        include_layout_cues: bool = True
    ):
        """Fine-tunes the cross-encoder."""
        train_ds = BoundaryPairDataset(train_boundaries, self.tokenizer, self.max_length, include_layout_cues)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        # Calculate class weights for imbalance
        labels = [int(b['label']) for b in train_boundaries]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=self.device) if n_pos > 0 else None
        
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
        
        print(f"Starting training for {epochs} epochs ({len(train_ds)} samples, {len(train_loader)} batches/epoch)...")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Check token_type_ids if available for model
                token_type_ids = batch.get('token_type_ids')
                kwargs = {}
                if token_type_ids is not None:
                    if hasattr(self.model, "config") and getattr(self.model.config, "type_vocab_size", 0) > 0:
                        kwargs['token_type_ids'] = token_type_ids.to(self.device)
                    
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            val_info = ""
            if val_boundaries:
                val_metrics = self.evaluate(val_boundaries, batch_size=batch_size, include_layout_cues=include_layout_cues)
                val_info = f" | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val Macro F1: {val_metrics['macro_f1']*100:.2f}%"
                
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}{val_info}")
            
        return self
        
    def predict_proba(
        self,
        boundaries: List[Dict[str, Any]],
        batch_size: int = 32,
        include_layout_cues: bool = True
    ) -> np.ndarray:
        """Returns class probabilities [P(SPLIT), P(JOIN)] for each boundary."""
        dataset = BoundaryPairDataset(boundaries, self.tokenizer, self.max_length, include_layout_cues)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                kwargs = {}
                if 'token_type_ids' in batch:
                    if hasattr(self.model, "config") and getattr(self.model.config, "type_vocab_size", 0) > 0:
                        kwargs['token_type_ids'] = batch['token_type_ids'].to(self.device)
                    
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
                
        return np.vstack(all_probs)
        
    def predict(
        self,
        boundaries: List[Dict[str, Any]],
        batch_size: int = 32,
        include_layout_cues: bool = True
    ) -> np.ndarray:
        """Returns binary predictions (1 = JOIN, 0 = SPLIT)."""
        probs = self.predict_proba(boundaries, batch_size=batch_size, include_layout_cues=include_layout_cues)
        return np.argmax(probs, axis=1)
        
    def evaluate(
        self,
        boundaries: List[Dict[str, Any]],
        batch_size: int = 32,
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
        cm = confusion_matrix(y, preds).tolist()
        
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "join_f1": float(join_f1),
            "split_f1": float(split_f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm
        }
        
    def save(self, output_dir: str):
        """Saves model and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")

    @classmethod
    def load(cls, model_path: str, device: Optional[str] = None, max_length: int = 128) -> "CrossEncoderClassifier":
        """Loads a saved cross-encoder model and tokenizer from disk."""
        return cls(model_name=model_path, device=device, max_length=max_length)

