from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import yaml
import os

# Load config
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Get model path from config
model_path = config["model"]["path"]
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move model to GPU if available
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_response(prompt: str) -> str:
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate text
    outputs = model.generate(**inputs, max_length=512, temperature=0.7, top_p=0.95)
    # Decode the generated tokens to text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    prompt = "Write a short story about a werewolf"
    response = generate_response(prompt)
    print(response)
