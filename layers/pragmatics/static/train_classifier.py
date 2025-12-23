# pragmatics/train_classifier.py
"""
Train a classifier to distinguish:
  - Class 0: NOT a save request (recall queries, casual chat, general questions)
  - Class 1: IS a save request (user providing info to remember)

Key insight: "Do you remember my name?" is a RECALL (class 0), not a save.
             "My name is John" is a SAVE (class 1).
"""
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import random

# -----------------------------------------------------------------
# GPU Detection
# -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# -----------------------------------------------------------------
# Create training data with proper class separation
# -----------------------------------------------------------------
print("Creating training dataset...")

# Import examples from separate files
from examples import SAVE_EXAMPLES, RECALL_EXAMPLES, CASUAL_EXAMPLES

# Combine all examples with proper labels
# Class 0 = NOT save (recall + casual)
# Class 1 = SAVE
texts = []
labels = []

# Add save examples (class 1)
for ex in SAVE_EXAMPLES:
    texts.append(ex)
    labels.append(1)

# Add recall examples (class 0) - KEY differentiators
for ex in RECALL_EXAMPLES:
    texts.append(ex)
    labels.append(0)

# Add casual examples (class 0)
for ex in CASUAL_EXAMPLES:
    texts.append(ex)
    labels.append(0)

# Balance the dataset
save_count = labels.count(1)
other_count = labels.count(0)
print(f"Before balancing: {save_count} save, {other_count} other")

# Replicate minority class to balance
if save_count < other_count:
    factor = (other_count // save_count)
    original_save = [(t, l) for t, l in zip(texts, labels) if l == 1]
    for _ in range(factor - 1):
        for t, l in original_save:
            texts.append(t)
            labels.append(l)
elif other_count < save_count:
    factor = (save_count // other_count)
    original_other = [(t, l) for t, l in zip(texts, labels) if l == 0]
    for _ in range(factor - 1):
        for t, l in original_other:
            texts.append(t)
            labels.append(l)

# Shuffle
combined = list(zip(texts, labels))
random.seed(42)
random.shuffle(combined)
texts, labels = zip(*combined)
texts = list(texts)
labels = list(labels)

print(f"After balancing: {labels.count(1)} save, {labels.count(0)} other")
print(f"Total dataset size: {len(texts)}")

# -----------------------------------------------------------------
# Tokenizer & train/val split
# -----------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Split texts and labels first
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# Then tokenize each split
train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=128
)

class MemDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# Create datasets
train_dataset = MemDataset({"input_ids": train_encodings["input_ids"], 
                             "attention_mask": train_encodings["attention_mask"]}, 
                            train_labels)
val_dataset = MemDataset({"input_ids": val_encodings["input_ids"], 
                           "attention_mask": val_encodings["attention_mask"]}, 
                          val_labels)

# -----------------------------------------------------------------
# Fine-tune DistilBERT
# -----------------------------------------------------------------
print("Initializing model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(device)  # Move model to GPU

# Increase batch size if GPU is available for faster training
batch_size = 32 if torch.cuda.is_available() else 16

training_args = TrainingArguments(
    output_dir="./distilbert_memory_output",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 2,
    num_train_epochs=3,
    learning_rate=5e-5,
    eval_strategy="epoch",  # Renamed from evaluation_strategy in newer transformers
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    fp16=torch.cuda.is_available(),  # Enable mixed precision on GPU
    dataloader_num_workers=0,  # Avoid issues on Windows
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# -----------------------------------------------------------------
# Save the checkpoint
# -----------------------------------------------------------------
save_path = Path(".")
model.save_pretrained(save_path / "distilbert_memory")
tokenizer.save_pretrained(save_path / "distilbert_memory")

print(f"✅ Model saved to {save_path / 'distilbert_memory'}")

# Test it with various examples
def test_model(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(logits, dim=-1).item()
    conf = probs[0][pred].item()
    label = "SAVE" if pred == 1 else "OTHER"
    return f"{label} ({conf:.2%})"

print("\n--- Quick test ---")
print("SAVE requests (should be SAVE):")
print(f"  'My name is John' -> {test_model('My name is John')}")
print(f"  'Remember my birthday is May 3rd' -> {test_model('Remember my birthday is May 3rd')}")
print(f"  'I work at Microsoft' -> {test_model('I work at Microsoft')}")

print("\nRECALL queries (should be OTHER):")
print(f"  'What is my name?' -> {test_model('What is my name?')}")
print(f"  'Do you remember my birthday?' -> {test_model('Do you remember my birthday?')}")
test_email = "What's my email?"
print(f"  '{test_email}' -> {test_model(test_email)}")

print("\nCasual chat (should be OTHER):")
print(f"  'How are you?' -> {test_model('How are you?')}")
print(f"  'What is the weather today?' -> {test_model('What is the weather today?')}")
print(f"  'Tell me a joke' -> {test_model('Tell me a joke')}")