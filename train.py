from models import model
from runtoken import tokenized_test, tokenized_train 
from transformers import TrainingArguments, Trainer


training_args = TrainingArguments(
    output_dir="./models/medical_qa",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test
)


trainer.train()