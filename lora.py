from models import model
from peft import LoraConfig, get_peft_model


# 1. Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q", "v"],
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)


# 2. Add LoRA to the pretrained model
model = get_peft_model(model, lora_config)


# 3. Check trainable parameters
model.print_trainable_parameters()