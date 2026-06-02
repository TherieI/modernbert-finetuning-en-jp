from transformers import pipeline
import time
import json
from datasets import load_dataset
import deepl

#import sample dataset with keigo
ds = load_dataset("ronantakizawa/japanese-honorifics")
print(ds)
sample = ds["train"]

# -------------------------
# 1. Load models
# -------------------------

# Japanese classifier
jp_classifier = pipeline(
    "text-classification",
    model="modernbert-politeness-ja"
)

# English classifier
en_classifier = pipeline(
    "text-classification",
    model="modernbert-politeness"
)

# -------------------------
# 2. Translation (simple version)
# -------------------------

from deep_translator import GoogleTranslator

def translate_jp_to_en(text):
    return GoogleTranslator(source='ja', target='en').translate(text)

def translate_en_to_jp(text):
    return GoogleTranslator(source='en', target='ja').translate(text)

deepl_translator = deepl.Translator("2a3c7a1a-d32c-4dda-b001-2604b55a8e81:fx")

def translate(text, source_language, translator): #source_language should be original text language, either 'ja' or 'en'
    if (translator == "google"):
        return GoogleTranslator(source=source_language)
    if (translator == "deepl"):
        return deepl_translator.translate_text(text, source_language, )
# -------------------------
# 3. Classification helpers
# -------------------------

def classify_japanese(text):
    output = jp_classifier(text, top_k=4)

    return {
        "input": text,
        "scores": output
    }

def classify_english(text):
    output = en_classifier(text, top_k=4)

    return {
        "input": text,
        "scores": output
    }

# -------------------------
# 4. Full pipeline
# -------------------------

def analyze_sentence(text):
    print("Running Japanese classification...")
    start = time.time()

    jp_result = classify_japanese(text)

    print(f"JP done in {time.time() - start:.2f}s")

    print("Translating...")
    translated = translate_jp_to_en(text)

    print("Running English classification...")
    start = time.time()

    en_result = classify_english(translated)

    print(f"EN done in {time.time() - start:.2f}s")

    print("Done.")

    return {
        "original_text": text,
        "translation": translated,
        "japanese_model": jp_result,
        "english_model": en_result
    }

def analyze_english_sentence(text):
    print("Running English classification...")
    start = time.time()

    en_result = classify_english(text)

    print(f"EN done in {time.time() - start:.2f}s")

    print("Translating...")
    translated = translate_en_to_jp(text)

    print("Running Japanese classification...")
    start = time.time()

    jp_result = classify_japanese(translated)

    print(f"JP done in {time.time() - start:.2f}s")

    print("Done.")

    return {
        "original_text": text,
        "translation": translated,
        "english_model": en_result,
        "japanese_model": jp_result
    }

#sample dataset processing
fields = ["base_sentence", "teineigo", "sonkeigo", "kenjogo"]

results = []
i = 1
for example in sample:
    print(f"Starting analysis {i}", flush=True)
    i += 1
    for field in fields:
        text = example[field]

        jp_result = classify_japanese(text)
        translated = translate_jp_to_en(text)
        en_result = classify_english(translated)

        results.append({
            "form": field,
            "jp_text": text,
            "translation": translated,
            "jp_scores": jp_result,
            "en_scores": en_result
        })

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
    