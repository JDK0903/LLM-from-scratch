import torch
from transformers import GPT2Tokenizer, GPT2Model

# Step 1: Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

# Step 2: Add special tokens and resize model
tokenizer.add_special_tokens({
    "pad_token": "<|pad|>",
    "bos_token": "<|startoftext|>",
    "eos_token": "<|endoftext|>"
})
model.resize_token_embeddings(len(tokenizer))

# Step 3: Load your text file
file_path = "C:\\Users\\USER\\sentence-t5-large IoTRepo\\pythonProject\\txt\\hawaii_wf_gpt2_finetune.txt"  # Change this to your actual path
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Step 4: Tokenize the text
inputs = tokenizer(text[:500], return_tensors="pt", truncation=True, max_length=512)

# Step 5: Get token embeddings (no gradient computation)
with torch.no_grad():
    outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state  # Shape: [1, seq_len, hidden_dim]

# Step 6: Use the embeddings
print("Shape:", token_embeddings.shape)
print("First token embedding:", token_embeddings[0, 0])  # Example
