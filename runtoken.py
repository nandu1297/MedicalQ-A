from split_data import train_dataset, validation_dataset, test_dataset
from transformers import AutoTokenizer


# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

print("Tokenizer loaded successfully")


# 2. Tokenize training data
tokenized_train = []

for example in train_dataset:

    # Convert question into token IDs
    question = tokenizer(
        example["question"],
        max_length=256,
        truncation=True
    )

    # Convert answer into token IDs
    answer = tokenizer(
        text_target=example["answer"],
        max_length=512,
        truncation=True
    )

    # Connect the answer with this question
    question["labels"] = answer["input_ids"]

    # Store the complete example
    tokenized_train.append(question)


print("Training data tokenized")


# 3. Tokenize validation data
tokenized_validation = []

for example in validation_dataset:

    question = tokenizer(
        example["question"],
        max_length=256,
        truncation=True
    )

    answer = tokenizer(
        text_target=example["answer"],
        max_length=512,
        truncation=True
    )

    question["labels"] = answer["input_ids"]

    tokenized_validation.append(question)


print("Validation data tokenized")


# 4. Tokenize test data
tokenized_test = []

for example in test_dataset:

    question = tokenizer(
        example["question"],
        max_length=256,
        truncation=True
    )

    answer = tokenizer(
        text_target=example["answer"],
        max_length=512,
        truncation=True
    )

    question["labels"] = answer["input_ids"]

    tokenized_test.append(question)


print("Test data tokenized")


# 5. Check one example
print("\nFirst tokenized training example:")
print(tokenized_train[0])