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
        Analyze sentiment for a list of texts.
        """
        # For a simple dashboard, sequential processing is fine. 
        # For production, we would use batch encoding.
        results = []
        for text in texts:
            try:
                # Handle empty or very short messages
                if not text or len(text.strip()) == 0:
                    results.append({"label": "Neutral", "score": 0.0, "scores": {"Negative": 0, "Neutral": 1, "Positive": 0}})
                    continue
                results.append(self.predict(text))
            except Exception as e:
                print(f"Error processing text: {text}. Error: {e}")
                results.append({"label": "Neutral", "score": 0.0, "scores": {"Negative": 0, "Neutral": 1, "Positive": 0}})
        return results
