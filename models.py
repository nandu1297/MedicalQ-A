from transformers import AutoModelForSeq2SeqLM


# Choose the pretrained model
model_name = "google/flan-t5-small"


# Load the pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


print("Model loaded successfully")
print(model)