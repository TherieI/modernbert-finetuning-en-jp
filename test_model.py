from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./modernbert-politeness"
)

text = "Enter three into the calculator and you should get your results"

result = classifier(text)

print(result)