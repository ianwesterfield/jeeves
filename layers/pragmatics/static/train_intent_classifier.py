# pragmatics/train_intent_classifier.py
"""
Train a 4-class intent classifier for Jeeves:
  - Class 0: CASUAL - General chat, greetings, questions
  - Class 1: SAVE - User providing info to remember  
  - Class 2: RECALL - User asking for remembered info
  - Class 3: TASK - Workspace/code/execution requests

This replaces the binary save/not-save classifier with proper 
multi-class intent detection.
"""
import torch
import random
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np

# -----------------------------------------------------------------
# GPU Detection
# -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# -----------------------------------------------------------------
# Intent Labels
# -----------------------------------------------------------------
INTENT_LABELS = {
    0: "casual",
    1: "save", 
    2: "recall",
    3: "task",
}

LABEL_TO_ID = {v: k for k, v in INTENT_LABELS.items()}

# -----------------------------------------------------------------
# Load Training Data
# -----------------------------------------------------------------
print("Loading training examples...")

from examples import SAVE_EXAMPLES, RECALL_EXAMPLES, CASUAL_EXAMPLES, TASK_EXAMPLES

print(f"  Casual: {len(CASUAL_EXAMPLES)} examples")
print(f"  Save: {len(SAVE_EXAMPLES)} examples")
print(f"  Recall: {len(RECALL_EXAMPLES)} examples")
print(f"  Task: {len(TASK_EXAMPLES)} examples")

# Combine all examples with labels
texts = []
labels = []

for ex in CASUAL_EXAMPLES:
    texts.append(ex)
    labels.append(LABEL_TO_ID["casual"])

for ex in SAVE_EXAMPLES:
    texts.append(ex)
    labels.append(LABEL_TO_ID["save"])

for ex in RECALL_EXAMPLES:
    texts.append(ex)
    labels.append(LABEL_TO_ID["recall"])

for ex in TASK_EXAMPLES:
    texts.append(ex)
    labels.append(LABEL_TO_ID["task"])

print(f"\nTotal examples: {len(texts)}")

# -----------------------------------------------------------------
# Balance Classes (oversample minority classes)
# -----------------------------------------------------------------
print("\nBalancing classes...")

class_counts = {}
for label in INTENT_LABELS.keys():
    class_counts[label] = labels.count(label)
    print(f"  {INTENT_LABELS[label]}: {class_counts[label]}")

max_count = max(class_counts.values())

# Oversample minority classes
balanced_texts = []
balanced_labels = []

for label_id, label_name in INTENT_LABELS.items():
    class_examples = [(t, l) for t, l in zip(texts, labels) if l == label_id]
    count = len(class_examples)
    
    # Add original examples
    for t, l in class_examples:
        balanced_texts.append(t)
        balanced_labels.append(l)
    
    # Oversample to match max class
    if count < max_count:
        oversample_count = max_count - count
        oversampled = random.choices(class_examples, k=oversample_count)
        for t, l in oversampled:
            balanced_texts.append(t)
            balanced_labels.append(l)

# Shuffle
combined = list(zip(balanced_texts, balanced_labels))
random.seed(42)
random.shuffle(combined)
texts, labels = zip(*combined)
texts = list(texts)
labels = list(labels)

print(f"\nAfter balancing: {len(texts)} total examples")
for label_id, label_name in INTENT_LABELS.items():
    print(f"  {label_name}: {labels.count(label_id)}")

# -----------------------------------------------------------------
# Tokenizer & Train/Val Split
# -----------------------------------------------------------------
print("\nTokenizing...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

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

# -----------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = IntentDataset(
    {"input_ids": train_encodings["input_ids"], 
     "attention_mask": train_encodings["attention_mask"]}, 
    train_labels
)
val_dataset = IntentDataset(
    {"input_ids": val_encodings["input_ids"], 
     "attention_mask": val_encodings["attention_mask"]}, 
    val_labels
)

# -----------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------
print("\nInitializing model...")

# Create label mappings for the model
id2label = INTENT_LABELS
label2id = LABEL_TO_ID

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4,
    id2label=id2label,
    label2id=label2id,
).to(device)

print(f"Model loaded with {model.num_labels} labels: {list(id2label.values())}")

# -----------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------
batch_size = 32 if torch.cuda.is_available() else 16
output_dir = "./distilbert_intent_output"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=f"{output_dir}/logs",
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
)

# -----------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# -----------------------------------------------------------------
# Training
# -----------------------------------------------------------------
print("\nStarting training...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# -----------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

eval_results = trainer.evaluate()
print(f"\nValidation Accuracy: {eval_results['eval_accuracy']:.4f}")

# Get predictions for detailed report
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

print("\nClassification Report:")
print(classification_report(val_labels, preds, target_names=list(INTENT_LABELS.values())))

print("\nConfusion Matrix:")
cm = confusion_matrix(val_labels, preds)
print(f"              {' '.join(f'{v:>8}' for v in INTENT_LABELS.values())}")
for i, row in enumerate(cm):
    print(f"{INTENT_LABELS[i]:>12}  {' '.join(f'{v:>8}' for v in row)}")

# -----------------------------------------------------------------
# Save Final Model
# -----------------------------------------------------------------
final_model_path = "./distilbert_intent"
print(f"\nSaving model to {final_model_path}...")

model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

# Save label mappings
import json
with open(f"{final_model_path}/intent_labels.json", "w") as f:
    json.dump({
        "id2label": id2label,
        "label2id": label2id,
    }, f, indent=2)

print("\n✓ Training complete!")
print(f"Model saved to: {final_model_path}")
print("\nTo use in production:")
print(f"  1. Copy {final_model_path}/ to the container")
print(f"  2. Set CLASSIFIER_MODEL=distilbert_intent")
print(f"  3. Restart pragmatics_api")
