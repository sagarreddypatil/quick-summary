import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bart = torch.hub.load("pytorch/fairseq", "bart.large.cnn")
bart.to(device)
bart.eval()
bart.half()


with open("sample.txt") as f:
    sample_text = f.read()

tokens = bart.encode(sample_text)
print(bart.decode(tokens))