
import json
from datasets import load_dataset
import deepl
from deep_translator import GoogleTranslator
import pandas as pd
from huggingface_hub import hf_hub_download
from datasets import Dataset, concatenate_datasets

deepl_translator = deepl.Translator("2a3c7a1a-d32c-4dda-b001-2604b55a8e81:fx")

# -------------------------
# 2. Translation (simple version)
# -------------------------

def translate_jp_to_en(text):
    return GoogleTranslator(source='ja', target='en').translate(text)

def translate_en_to_jp(text):
    return GoogleTranslator(source='en', target='ja').translate(text)

def translate_jp_to_en_deepl(text):
    return deepl_translator.translate_text(text, source_lang="ja", target_lang="EN-US")

def translate_en_to_jp_deepl(text):
    return deepl_translator.translate_text(text, source_lang="en", target_lang="ja")

def stanford_politeness():
    # target is the target number of each type of sentence.
    TARGET = 100

    # download a specific file
    file_path = hf_hub_download(
        repo_id="Cleanlab/stanford-politeness",
        filename="fine-tuning/train.csv",
        repo_type="dataset"
    )
    df = pd.read_csv(file_path)
    full_dataset = Dataset.from_pandas(df)

    subsets = []
    # extract a TARGET amount of polite, impolite, and neutral sentences
    for level in ["polite", "neutral", "impolite"]:
        subsets.append(full_dataset.filter(lambda content: content["completion"] == level).select(range(TARGET)))
    dataset = concatenate_datasets(subsets)

    updated = []
    for i, entry in enumerate(dataset):
        try:
            gt = translate_en_to_jp(entry["prompt"])
            dt = translate_en_to_jp_deepl(entry["prompt"])
            if dt == type(list):
                dt = dt[0]
            print(f"{[i]}. '{entry["prompt"]}' completed!")
        except:
            print(f"'{entry["prompt"]}' ({entry["completion"]}) couldnt be translated")

        updated.append({
            "sentiment": entry["completion"],
            "en": entry["prompt"],
            "jp_google_translate": gt,
            "jp_deep_translate": str(dt.text)
        })

    with open(f"datasets/stanford_politeness_translations.json", "w", encoding='utf-8') as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

def ronantakizawa():
    #import sample dataset with keigo
    ds = load_dataset("ronantakizawa/japanese-honorifics")

    #sample dataset processing
    politeness = ["base_sentence", "teineigo", "sonkeigo", "kenjogo"]

    results = []
    for i, content in enumerate(ds["train"]):
        for level in politeness:
            try:
                gt = translate_jp_to_en(content[level])
                dl = translate_jp_to_en_deepl(content[level])
                print(f"[{i}][{level}] Translation of {content[level]} complete!")
            except:
                print(f"[ERROR] Failed to translate: '{content[level]}'")

            results.append({
                "level": level,
                "jp": content[level],
                "en_gt": gt,
                "en_dl": str(dl.text)
            })

    with open(f"datasets/ronantakizawa_translations.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    ronantakizawa()

if __name__ == "__main__":
    main()