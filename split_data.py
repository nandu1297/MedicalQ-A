from clean_data import clean_dataset


# Step 1: Separate 10% for testing
test_split = clean_dataset.train_test_split(
    test_size=0.1,
    seed=42
)


# Step 2: Split the remaining 90%
# into 80% train and 10% validation
train_val_split = test_split["train"].train_test_split(
    test_size=1/9,
    seed=42
)


# Final datasets
train_dataset = train_val_split["train"]
validation_dataset = train_val_split["test"]
test_dataset = test_split["test"]


# Check sizes
print("Total:", len(clean_dataset))
print("Train:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Test:", len(test_dataset))