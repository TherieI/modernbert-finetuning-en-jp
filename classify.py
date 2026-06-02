import json
from transformers import pipeline
from sentiment import calc_score

EN_MODEL_NAME = "modernbert-politeness"
JP_MODEL_NAME = "modernbert-politeness-ja"


def get_classifiers():
    classifier_en = pipeline(
        "text-classification",
        model=f"./{EN_MODEL_NAME}"
    )
    classifier_jp = pipeline(
        "text-classification",
        model=f"./{JP_MODEL_NAME}"
    )
    return classifier_en, classifier_jp

def stanford_politeness():
    (classifier_en, classifier_jp) = get_classifiers()

    results = []
    with open("stanford_politeness_sentences_all.json", "r", encoding="utf-8") as f:
        for i, content in enumerate(json.load(f)):

            en_class = classifier_en(content["en"], top_k=None)
            jp_class_gt = classifier_jp(content["jp_google_translate"], top_k=None)
            jp_class_dl = classifier_jp(content["jp_deep_translate"], top_k=None)

            # EDIT IF NECESSARY SO 0 is IMPOLITE, 1 is POLITE
            jp_score_gt = 1 - calc_score(jp_class_gt)
            jp_score_dl = 1 - calc_score(jp_class_dl)
            en_score = calc_score(en_class)

            results.append({
                    "sentiment": content["sentiment"],
                    "en": content["en"],
                    "en_labels": en_class,
                    "en_score": en_score,
                    "jp": [
                        {
                            "translator": "google translate",
                            "translation": content["jp_google_translate"],
                            "labels": jp_class_gt,
                            "score": jp_score_gt,
                            "error": abs(jp_score_gt - en_score)
                        }, 
                        {
                            "translator": "deepl",
                            "translation": content["jp_deep_translate"],
                            "labels": jp_class_dl,
                            "score": jp_score_dl,
                            "error": abs(jp_score_dl - en_score)
                        }
                    ]
                })

            print(f"[{i}] Analysis complete")

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def ronantakizawa():
    (classifier_en, classifier_jp) = get_classifiers()

    results = []
    with open("ronantakizawa_sentences.json", "r", encoding="utf-8") as f:
        for i, content in enumerate(json.load(f)):

            jp_class = classifier_jp(content["jp"], top_k=None)
            en_class_gt = classifier_en(content["en_gt"], top_k=None)
            en_class_dl = classifier_en(content["en_dl"], top_k=None)

            # EDIT IF NECESSARY SO 0 is IMPOLITE, 1 is POLITE
            en_score_gt = calc_score(en_class_gt)
            en_score_dl = calc_score(en_class_dl)
            jp_score = 1 - calc_score(jp_class)

            results.append({
                    "level": content["level"],
                    "jp": content["jp"],
                    "jp_labels": jp_class,
                    "jp_score": jp_score,
                    "en": [
                        {
                            "translator": "google translate",
                            "translation": content["en_gt"],
                            "labels": en_class_gt,
                            "score": en_score_gt,
                            "error": abs(en_score_gt - jp_score)
                        }, 
                        {
                            "translator": "deepl",
                            "translation": content["en_dl"],
                            "labels": en_class_dl,
                            "score": en_score_dl,
                            "error": abs(en_score_dl - jp_score)
                        }
                    ]
                })

            print(f"[{i}] Analysis complete")

    with open("ronantakizawa_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    ronantakizawa()

if __name__ == "__main__":
    main()