import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load your fine-tuned chatbot model
model_path = "./gpt2-maui-chatbot"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("🤖 Chatbot is ready! Type your question (or 'exit' to quit)")

while True:
    user_input = input("\nQ: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = f"<|startoftext|> Q: {user_input} A:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "A:" in generated_text:
        answer = generated_text.split("A:")[1].strip()
    else:
        answer = generated_text.strip()

    # Heuristic check for fallback
    if len(answer.split()) < 5 or answer.lower() in ["i don't know", "not sure", "unknown"]:
        print("A: Answer not found.")
    else:
        print(f"A: {answer}")
