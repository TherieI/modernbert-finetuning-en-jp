from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "answerdotai/ModernBERT-base"
TRAINED_MODEL_NAME = "modernbert-politeness-en"

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }


def main():
    dataset = load_dataset("frfede/politeness-corpus")
    dataset = dataset.remove_columns(["sentiment"])
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenize = lambda example: tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    tokenized_dataset = dataset.map(tokenize)

    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # Set PyTorch format
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    split_dataset = tokenized_dataset["train"].train_test_split(
        test_size=0.15,
        seed=42
    )

    print(split_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    training_args = TrainingArguments(
        output_dir="./results",

        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        num_train_epochs=3,

        weight_decay=0.01,

        logging_dir="./logs",

        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],

        processing_class=tokenizer,

        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(f"./{TRAINED_MODEL_NAME}")
    tokenizer.save_pretrained(f"./{TRAINED_MODEL_NAME}")

    

if __name__ == "__main__":
    main()