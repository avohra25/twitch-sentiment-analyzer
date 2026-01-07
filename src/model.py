from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
import torch

class SentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize the sentiment analyzer with a RoBERTa model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ['Negative', 'Neutral', 'Positive']

    def predict(self, text):
        """
        Analyze sentiment of a single text string.
        Returns a dictionary with 'label' and 'score'.
        """
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        
        top_label = self.labels[ranking[0]]
        top_score = scores[ranking[0]]
        
        return {
            "label": top_label,
            "score": float(top_score),
            "scores": {
                "Negative": float(scores[0]),
                "Neutral": float(scores[1]),
                "Positive": float(scores[2])
            }
        }

    def predict_batch(self, texts):
        """
        Analyze sentiment for a list of texts using vectorized inference.
        """
        if not texts:
            return []
            
        # Replace empty strings to keep alignment and avoid errors
        clean_texts = [t if t and t.strip() else "." for t in texts]
        
        # Tokenize batch
        encoded_input = self.tokenizer(clean_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        scores = output.logits.detach().numpy()
        scores = softmax(scores, axis=1)
        
        results = []
        for i, text in enumerate(texts):
            # If original was effectively empty, return neutral
            if not text or not text.strip():
                 results.append({"label": "Neutral", "score": 0.0, "scores": {"Negative": 0, "Neutral": 1, "Positive": 0}})
                 continue
            
            s = scores[i]
            ranking = np.argsort(s)[::-1]
            top_label = self.labels[ranking[0]]
            top_score = s[ranking[0]]
            
            results.append({
                "label": top_label,
                "score": float(top_score),
                "scores": {
                    "Negative": float(s[0]),
                    "Neutral": float(s[1]),
                    "Positive": float(s[2])
                }
            })
        return results
