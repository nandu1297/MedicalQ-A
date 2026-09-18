from lora import model
from runtoken import tokenized_test, tokenizer
from transformers import Trainer, DataCollatorForSeq2Seq


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)


trainer = Trainer(
    model=model,
    data_collator=data_collator
)


results = trainer.evaluate(
    eval_dataset=tokenized_test
)


print("Evaluation Results:")
print(results)