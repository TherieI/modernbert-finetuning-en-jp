from typing import Any

from transformers import pipeline
from argparse import ArgumentParser

def calc_score(result: list[dict[str, Any]]) -> float:
    total = 0.0
    for i, score in enumerate(sorted(result, key=lambda content: int(content['label'][-1]))):
        total += score['score'] * (i + 1)
    return (total - 1) / len(result)

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

    print(result)
    print(sorted(result, key=lambda content: int(content['label'][-1])))
    print(calc_score(result))

if __name__ == "__main__":
    main()