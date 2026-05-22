from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

MODEL_NAME = "sbintuitions/modernbert-ja-130m"
TRAINED_MODEL_NAME = "modernbert-politeness-ja"

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }


def main():
    # features = Features({
    #     "例文": Value("large_string"),
    #     "レベル": Value("int8"),
    #     "尊敬語": Value("int8"),
    #     "謙譲語": Value("int8"),
    #     "丁寧語": Value("int8"),
    #     "フィールド": Value("large_string"),
    # })
    # dataset = load_dataset(
    #     "csv",
    #     "keico_corpus.csv",
    #     encoding="utf-8"
    # )

    ds = pd.read_csv("keico_corpus.csv", encoding="utf-8")
    ds.columns = ds.columns.str.strip()
    ds = ds.rename(columns={
        "例文": "text",
        "レベル": "labels"
    })
    ds = ds[["text", "labels"]]
    dataset = Dataset.from_pandas(ds)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenize = lambda example: tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    tokenized_dataset = dataset.map(tokenize)

    # Set PyTorch format
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    print(tokenized_dataset)

    split_dataset = tokenized_dataset.train_test_split(
        test_size=0.15,
        seed=42
    )

    print(split_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5
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