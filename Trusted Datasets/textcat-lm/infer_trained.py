import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# setup seed
torch.manual_seed(0)

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model and tokenizer
model_path = "./models/trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

# Define the labels corresponding to the levels
labels = [
    "Elementary facts",
    "Identification",
    "Skill assessment",
    "Localization",
    "Explanation",
    "Reflection",
    "Critique",
    "Recommend recourse"
]

# Function to perform inference
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.softmax(logits, dim=-1)
    predicted_label_id = torch.argmax(predictions, dim=-1).item()
    confidence_score = torch.max(predictions).item()
    return labels[predicted_label_id], confidence_score

# Sample texts for inference
sample_texts = [
    "The surgeon used a scalpel to make the initial incision.",
    "The procedure identified the heart and surrounding arteries.",
    "The surgeon demonstrated excellent skill during the procedure."
]

# Perform inference on sample texts
for text in sample_texts:
    label, confidence = classify_text(text)
    print(f"{text[:20]}... -> {label} (confidence: {confidence:.4f})")
