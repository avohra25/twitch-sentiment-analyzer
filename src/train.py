import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

def train(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", output_dir="./models/twitch_roberta", epochs=1):
    print(f"Loading dataset 'lparkourer10/twitch_chat'...")
    # Note: 'lparkourer10/twitch_chat' might have specific splits or column names. 
    # We assume 'text' and 'label' columns exist or need mapping.
    dataset = load_dataset("lparkourer10/twitch_chat")
    
    # Check column names
    print(f"Dataset features: {dataset['train'].features}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    print("Tokenizing dataset...")
    # Assuming the dataset has a 'text' column. 
    # Current dataset viewer shows 'message_content' and 'sentiment' (0,1,2 probably?)
    # We might need to rename columns or adjust. 
    # Let's inspect first dynamic in real usage, but here's a generic map:
    
    # Mapping for lparkourer10/twitch_chat based on HF preview:
    # content: string
    # sentiment: int (0=Negative, 1=Neutral, 2=Positive) - Verify this mapping!
    
    # Adjust column names if needed
    if 'message_content' in dataset['train'].column_names:
        dataset = dataset.rename_column("message_content", "text")
    if 'sentiment' in dataset['train'].column_names:
        dataset = dataset.rename_column("sentiment", "label")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Split
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000)) # Small subset for demo
    eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000, 1200))

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        feature_mode="classification",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        no_cuda=not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        use_mps_device=torch.backends.mps.is_available() # optimize for Mac
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    train(epochs=args.epochs)
