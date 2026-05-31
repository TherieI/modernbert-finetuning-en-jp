from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "answerdotai/ModernBERT-base"
TRAINED_MODEL_NAME = "modernbert-politeness-en"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

def preprocess(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding=True, max_length=128)
    tokenized["labels"] = [float(label) for label in examples["label"]]
    return tokenized

def parse_dataset() -> DatasetDict:
    dataset = load_dataset("frfede/politeness-corpus")
    dataset = dataset.remove_columns(["sentiment"])

    return dataset.map(preprocess, batched=True)


def main():
    tokenized_dataset = parse_dataset()

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
        num_labels=1
    )

    training_args = TrainingArguments(
       output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,          # helps early training stability
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],

        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(f"./{TRAINED_MODEL_NAME}")
    tokenizer.save_pretrained(f"./{TRAINED_MODEL_NAME}")

    

if __name__ == "__main__":
    main()