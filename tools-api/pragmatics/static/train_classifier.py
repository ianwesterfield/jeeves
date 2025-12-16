# pragmatics/train_classifier.py
import torch
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# -----------------------------------------------------------------
# 🚀 GPU Detection
# -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# -----------------------------------------------------------------
# 1️⃣ Load a real dataset from Hugging Face
# -----------------------------------------------------------------
# Using the "silicone" dataset which has dialogue acts
# You can also use "swda" (Switchboard) or create your own
print("Loading dataset from Hugging Face...")
# Option 1: Use silicone with trust_remote_code (legacy loader)
# dataset = load_dataset("silicone", "dyda_da", split="train[:2000]", trust_remote_code=True)

# Option 2: Use emotion dataset and augment with memory-save examples
from datasets import Dataset
import random

# Load emotion dataset for diversity
emotion_data = load_dataset("dair-ai/emotion", split="train")

# Create synthetic memory-save examples (class 1)
save_examples = [
    # Original list:
    "Remember my name is Sarah",
    "Can you save that my birthday is May 3rd",
    "Please remember I live in Seattle",
    "Don't forget my email is john@example.com",
    "Keep in mind I prefer vegetarian options",
    "Note that I work at Microsoft",
    "Remember my phone number is 555-1234",
    "Save the fact that I have two cats",
    "Please keep track of my appointment on Tuesday",
    "Can you remember my wife's name is Emma",
    "Store this: my favorite color is blue",
    "Remember I'm allergic to peanuts",
    "Keep this in mind: I speak Spanish fluently",
    "Note down that my anniversary is June 15",
    "Save that I drive a Tesla Model 3",
    "Remember my doctor's appointment is at 3pm",
    "Can you keep track that I'm training for a marathon",
    "Please remember my son goes to Lincoln Elementary",
    "Don't forget I need to call the bank tomorrow",
    "Keep in mind my meeting with Sarah next Monday",
    
    # Colloquial variations:
    "Hey, can you just save this: my name's Sarah?",
    "Save that my birthday is May 3rd, yeah?",
    "Please don't forget I live in Seattle, okay?",
    "Got an email address? Mine's john@example.com",
    "Just remember I'm a veggie lover, dude!",
    "Note down my phone number: it's 555-1234",
    "Save the fact that I've got two fur babies at home",
    "Keep track of my appointment on Tuesday, okay?",
    "Can you just remember my wife's name is Emma?",
    "Store this: my favorite color is blue, bro!",
    "Remember I'm super allergic to peanuts, man!",
    "Just keep in mind that I speak Spanish fluently, bro",
    "Save the date: my anniversary is June 15th",
    "Hey, just remember I drive a Tesla Model 3",
    "Got an appointment with my doctor at 3pm?",
    "Can you just keep track that I'm training for a marathon?",
    "Please don't forget I need to call the bank tomorrow, okay?",
    "Just keep in mind I've got a meeting with Sarah next Monday",
    
    # Vague directives:
    "Save this stuff for me",
    "Remember, like, everything about me",
    "Keep track of my schedule or something",
    "Note down whatever's important, thanks",
    "Save the essentials, man",
    "Just remember, uh, details are important too",
    "Save the date and time for my appointment",
    "Keep in mind I'm running a marathon soon",
    "Remember to call the bank tomorrow",
    "Save the fact that I've got two cats at home",
    
    # Idiomatic expressions:
    "Save that I'm a total bookworm, dude!",
    "Remember I'm allergic to peanuts, big time!",
    "Note down my favorite color: it's blue, no question!",
    "Save the fact that I speak Spanish like a native",
    "Remember I'm training for a marathon, every step counts!",
    
    # Dialects:
    "Yo, save this: my birthday's on May 3rd, word?",
    "Hey, can you just keep track that I'm from Seattle, G?" 
    "Save the fact that I'm allergic to peanuts, for real",
    
    # Tone and style variations:
    "Remember, it's really important to me",
    "Just save this: my favorite color is blue",
    "Keep in mind I've got a meeting with Sarah next Monday, yeah?",
    "Save the date and time for my appointment, man",
    "Don't forget, like, everything about me, okay?"
    
    # More examples that avoid direct references to saving or memorizing information:
    "Just so you know, my name's Sarah",
    "I've got a lot going on this week, can you help me get organized?",
    "My wife's name is Emma, just FYI",
    "If I don't call the bank tomorrow, it'll be a mess!",
    "Got any free time? I need to schedule an appointment",
    "I'm trying to remember everything for my marathon training",
    "Can you help me set reminders for my upcoming meetings?",
    "My favorite color is blue, but I also love purple",
    "Just a heads up, my phone number's 555-1234",
    "If you could keep this under your hat, that'd be great",
    "Save the date: my anniversary is June 15th (don't forget to get me a gift)",
    "I'm trying to remember everything about my cat, Luna",
    "Keep in mind I've got two fur babies at home (they're adorable)",
    "If you could help me keep track of my schedule, that'd be awesome",
    "Just so you know, I work at Microsoft",
    "My doctor's appointment is at 3pm tomorrow, don't forget to remind me",
    "I'm training for a marathon and need all the support I can get"
] * 25  # Replicate to balance dataset

# Create labels: emotion examples are class 0 (other), save examples are class 1 (remember)
texts = [ex["text"] for ex in emotion_data] + save_examples
labels = [0] * len(emotion_data) + [1] * len(save_examples)

# Shuffle together
combined = list(zip(texts, labels))
random.shuffle(combined)
texts, labels = zip(*combined)

# Create dataset (already have texts and labels from above)
texts = list(texts)
labels = list(labels)
print(f"Created dataset with {len(texts)} examples")
print(f"Class distribution: {labels.count(0)} other, {labels.count(1)} remember")

# -----------------------------------------------------------------
# 2️⃣ Tokenizer & train/val split
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
# 3️⃣ Fine-tune DistilBERT
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
# 4️⃣ Save the checkpoint
# -----------------------------------------------------------------
save_path = Path(".")
model.save_pretrained(save_path / "distilbert_memory")
tokenizer.save_pretrained(save_path / "distilbert_memory")

print(f"✅ Model saved to {save_path / 'distilbert_memory'}")

# Test it
def test_model(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    model.eval()  # Ensure eval mode
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    return "Remember" if pred == 1 else "Other"

print("\n--- Quick test ---")
print(f"'Remember my birthday is May 3rd' -> {test_model('Remember my birthday is May 3rd')}")
print(f"'What's the weather today?' -> {test_model('What is the weather today?')}")