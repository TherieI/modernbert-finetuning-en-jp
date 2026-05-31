from transformers import pipeline
from argparse import ArgumentParser

def main():
    parser = ArgumentParser("test_model")
    parser.add_argument("model", help="name of the model")
    parser.add_argument("input", help="example sentence to parse")

    args = parser.parse_args()

    classifier = pipeline(
        "text-classification",
        model=f"./{args.model}"
    )

    result = classifier(args.input)

    print(result)

if __name__ == "__main__":
    main()