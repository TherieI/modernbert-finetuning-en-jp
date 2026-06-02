from typing import Any

from transformers import pipeline
from argparse import ArgumentParser

def calc_score(result: list[dict[str, Any]]) -> float:
    total = 0.0
    for i, score in enumerate(sorted(result, key=lambda content: int(content['label'][-1]))):
        total += score['score'] * i
    return total / (len(result) - 1)

def main():
    parser = ArgumentParser("test_model")
    parser.add_argument("model", help="name of the model")
    parser.add_argument("input", help="example sentence to parse")

    args = parser.parse_args()

    classifier = pipeline(
        "text-classification",
        model=f"./{args.model}"
    )

    result = classifier(args.input, top_k=None)

    print(sorted(result, key=lambda content: int(content['label'][-1])))

    s = calc_score(result)
    s = 1-s if args.model == "modernbert-politeness-ja" else s
    print(s)

if __name__ == "__main__":
    main()