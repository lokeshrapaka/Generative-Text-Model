
# Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set model to evaluation mode
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Text generation function
def generate_text(prompt, max_length=100, temperature=0.7):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example prompt
prompt = "Artificial Intelligence is"
generated = generate_text(prompt)
print("🔮 Generated Output:\n")
print(generated)
