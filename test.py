from lora import model
from runtoken import tokenizer

question = "What are the symptoms of diabetes?"

inputs = tokenizer(
    question,
    return_tensors="pt",
    max_length=256,
    truncation=True
)

outputs = model.generate(
    **inputs,
    max_length=150
)


answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("Question:", question)
print("Answer:", answer)