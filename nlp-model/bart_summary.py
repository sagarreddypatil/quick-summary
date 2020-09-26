from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import time

with open("sample.txt") as f:
    sample_text = f.read()

model_load_start = time.time()

model = AutoModelWithLMHead.from_pretrained("sshleifer/distilbart-cnn-6-6")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6")

print(f"Model loaded in {round(time.time() - model_load_start, 2)}s")

inputs = tokenizer.encode(
    sample_text, return_tensors="pt", max_length=1024, truncation=True
)
summarize_start = time.time()
outputs = model.generate(
    input_ids=inputs,
    attention_mask=torch.IntTensor([[1] * len(inputs[0])]),
    max_length=100,
    min_length=25,
    length_penalty=1.0,
    num_beams=3,
    early_stopping=True,
)
output_text = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
reduction = 1 - len(output_text) / len(sample_text)
print(
    f"---------------------------------Summarization took {round(time.time() - summarize_start, 2)}s at {round(reduction, 4) * 100}% reduction---------------------------------"
)
print(output_text)
