import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    LineByLineTextDataset,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import os

# ✅ Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ✅ Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Add and resize for special tokens
tokenizer.add_special_tokens({
    "pad_token": "<|pad|>",
    "bos_token": "<|startoftext|>",
    "eos_token": "<|endoftext|>"
})
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# ✅ Paths to your text files
paragraph_path = "C:\\Users\\USER\\sentence-t5-large IoTRepo\\pythonProject\\txt\\hawaii_wf_gpt2_finetune.txt"
qa_path = "C:\\Users\\USER\\sentence-t5-large IoTRepo\\pythonProject\\txt\\hawaii_wf_qa_finetune.txt"

# ✅ Helper to load dataset from line-by-line file
def load_dataset(file_path, tokenizer, block_size=128):
    return LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

# ✅ Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT uses causal LM
)

# ✅ Training argument factory
def get_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        num_train_epochs=3,
        save_steps=200,
        save_total_limit=2,
        logging_steps=10,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",  # Disable WandB
        prediction_loss_only=True
    )

# ✅ Stage 1: Paragraph pretraining
print("\n🔧 Stage 1: Fine-tuning on paragraph data...")
paragraph_dataset = load_dataset(paragraph_path, tokenizer, block_size=512)
trainer_paragraph = Trainer(
    model=model,
    args=get_args("./gpt2-paragraph"),
    data_collator=data_collator,
    train_dataset=paragraph_dataset,
)
trainer_paragraph.train()

# ✅ Stage 2: Q&A fine-tuning
print("\n🎯 Stage 2: Fine-tuning on Q&A data...")
qa_dataset = load_dataset(qa_path, tokenizer, block_size=128)
print(f"📏 QA Dataset has {len(qa_dataset)} samples.")

trainer_qa = Trainer(
    model=model,
    args=get_args("./gpt2-qa"),
    data_collator=data_collator,
    train_dataset=qa_dataset,
)
trainer_qa.train()

# ✅ Save final model
final_model_path = "./gpt2-maui-chatbot"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"\n✅ Final model saved at: {final_model_path}")
