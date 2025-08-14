import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from typing import List, Dict, Any
import re
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab

class FinancialNER:
    def __init__(self, model_name="yiyanghkust/finbert-fls"):
        """
        Initialize the Financial NER service.
        
        Args:
            model_name: Name of the pre-trained model to use. Default is 'yiyanghkust/finbert-fls'
                       which is a Financial Language Similarity model fine-tuned on financial statements.
        """
        self.device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        # Initialize spaCy for rule-based NER as fallback
        self.nlp = spacy.load("en_core_web_sm")
        self._add_financial_patterns()
        
        # Initialize the NER pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=self.device
        )
    
    def _add_financial_patterns(self):
        """Add financial entity patterns to spaCy's matcher."""
        # Add patterns for common financial entities
        ruler = self.nlp.add_pipe("entity_ruler")
        patterns = [
            {"label": "CURRENCY", "pattern": [{"TEXT": {"REGEX": r"^\$?\d+(\.\d{1,2})?$"}}]},
            {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"}}]},
            {"label": "ACCOUNT_NUMBER", "pattern": [{"TEXT": {"REGEX": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"}}]},
            {"label": "PERCENT", "pattern": [{"TEXT": {"REGEX": r"\d+%"}}]},
        ]
        ruler.add_patterns(patterns)
    
    def process_receipt_text(self, ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process OCR results to extract financial entities.
        
        Args:
            ocr_results: List of OCR results with text and position information
            
        Returns:
            List of entities with their types, text, positions, and confidence scores
        """
        # Combine all OCR text for processing
        full_text = " ".join([res['value']['text'][0] for res in ocr_results])
        
        # Get NER predictions
        ner_results = self.ner_pipeline(full_text)
        
        # Process results to match the OCR positions
        entities = []
        for entity in ner_results:
            entity_text = entity['word']
            entity_type = entity['entity_group']
            confidence = entity['score']
            
            # Find matching OCR result for position information
            for ocr_result in ocr_results:
                ocr_text = ocr_result['value']['text'][0]
                if entity_text.lower() in ocr_text.lower():
                    entities.append({
                        'text': entity_text,
                        'type': entity_type,
                        'score': confidence,
                        'position': {
                            'x': ocr_result['value']['x'],
                            'y': ocr_result['value']['y'],
                            'width': ocr_result['value']['width'],
                            'height': ocr_result['value']['height']
                        }
                    })
                    break
        
        return entities
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from plain text using both transformer and rule-based methods.
        
        Args:
            text: Input text to process
            
        Returns:
            List of entities with their types and confidence scores
        """
        # Get transformer-based NER results
        ner_results = self.ner_pipeline(text)
        
        # Get rule-based NER results
        doc = self.nlp(text)
        
        # Combine and deduplicate results
        entities = []
        processed_text = set()
        
        # Add transformer-based entities
        for entity in ner_results:
            entity_text = entity['word']
            if entity_text not in processed_text:
                entities.append({
                    'text': entity_text,
                    'type': entity['entity_group'],
                    'score': float(entity['score']),
                    'method': 'transformer'
                })
                processed_text.add(entity_text)
        
        # Add rule-based entities
        for ent in doc.ents:
            if ent.text not in processed_text and ent.label_ in ['DATE', 'MONEY', 'CARDINAL', 'PERCENT', 'ORG', 'GPE']:
                entities.append({
                    'text': ent.text,
                    'type': ent.label_,
                    'score': 1.0,  # Rule-based has max confidence
                    'method': 'rule_based'
                })
                processed_text.add(ent.text)
        
        return entities
