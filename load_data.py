from datasets import load_dataset



dataset = load_dataset("lavita/MedQuaD")

print(dataset)
train_data =dataset["train"]
