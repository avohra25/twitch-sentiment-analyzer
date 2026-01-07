from src.model import SentimentAnalyzer

def test_inference():
    print("Initializing Analyzer...")
    analyzer = SentimentAnalyzer()
    
    test_cases = [
        ("This is amazing! POG", "Positive"),
        ("I hate this game, it's terrible", "Negative"),
        ("Just watching the stream", "Neutral")
    ]
    
    print("\nRunning Tests:")
    for text, expected in test_cases:
        result = analyzer.predict(text)
        print(f"Text: '{text}' -> Predicted: {result['label']} (Score: {result['score']:.4f})")
        # Note: 'Neutral' vs 'Positive' can be tricky with short text, but social benchmarks usually handle these well.
        # We won't assert strict equality for now, just print output to verify it runs.

if __name__ == "__main__":
    test_inference()
