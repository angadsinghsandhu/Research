import torch
from transformers import pipeline

# setup seed
torch.manual_seed(0)

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setup classifier
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=device)
sequences = [
    "The surgeon used a scalpel to make the initial incision.",
    "The procedure identified the heart and surrounding arteries.",
    "The surgeon demonstrated excellent skill during the procedure."
]
labels = [ "Elementary facts", "Identification", "Skill assessment",
    "Localization", "Explanation", "Reflection", "Critique", "Recommend recourse" ]

for sequence in sequences:
    output = classifier(sequence, labels, multi_label=False)
    max_score = max(output['scores'])
    max_score_label = output['labels'][output['scores'].index(max_score)]
    print(f"{sequence[:20]}... -> {max_score_label} (confidence: {max_score:.4f})")