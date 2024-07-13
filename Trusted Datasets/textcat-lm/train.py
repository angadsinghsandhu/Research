from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import numpy as np
import evaluate, torch

# setup seed
torch.manual_seed(0)

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset('json', data_files={'train': './data/train.json', 'test': './data/test.json'})

# Display dataset structure
print(dataset)

# Load tokenizer and model
model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.06,
    fp16=True
)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8, ignore_mismatched_sizes=True).to(device)

# # Resize model's classification head to match the number of labels
# model.classifier = torch.nn.Linear(model.config.hidden_size, 8)
# model.num_labels = 8

# Define compute metrics function
metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
results = trainer.evaluate()
print(results)

# Save model and tokenizer
model.save_pretrained("./models/trained_model")
tokenizer.save_pretrained("./models/trained_model")
