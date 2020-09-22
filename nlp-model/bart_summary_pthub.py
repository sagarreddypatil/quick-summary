import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model_load_start = time.time()

bart = torch.hub.load("pytorch/fairseq", "bart.large.cnn")
if device.type == "cuda":
    bart.half()
bart.to(device)
bart.eval()

print(f"Model loaded in {round(time.time() - model_load_start, 2)}s")

count = 1
bsz = 32

with open("sample.txt") as source:
    sample_text = source.read()

summarize_start = time.time()
summary = bart.sample(
    [sample_text],
    beam=3,
    lenpen=1.0,
    max_len_b=100,
    min_len=25,
    no_repeat_ngram_size=3,
)[0]

reduction = 1 - len(summary) / len(sample_text)

print(
    f"---------------------------------Summarization took {round(time.time() - summarize_start, 2)}s at {round(reduction, 4) * 100}% reduction---------------------------------"
)
print(summary)