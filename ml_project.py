import re
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================================
# 1. Text Cleaning Function
# ============================================================================

def clean_text(text: str) -> str:
    """
    Cleans the tweet text:
      - Removes URLs
      - Removes mentions (@username)
      - Removes extra whitespace
      - Removes some punctuation artifacts
    You can expand this function with additional cleaning steps.
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)         # remove URLs
    text = re.sub(r'@\w+', '', text)              # remove @mentions
    text = re.sub(r'#', '', text)                # remove hashtag symbol (keep the word)
    text = re.sub(r'\s+', ' ', text).strip()      # remove extra spaces
    return text

# ============================================================================
# 2. Dataset Class for PyTorch
# ============================================================================

class DisasterDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        item = {key: encoding[key].squeeze(0) for key in encoding}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ============================================================================
# 3. Compute Metrics for Evaluation
# ============================================================================

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# ============================================================================
# 4. Load and Prepare Data
# ============================================================================

# Change the file paths if needed
df_train = pd.read_csv('train.csv', dtype={'id': np.int32, 'target': np.int8})
df_test = pd.read_csv('test.csv', dtype={'id': np.int32})

# Apply cleaning (if desired, you can combine text with keyword/location features)
df_train['text_clean'] = df_train['text'].apply(clean_text)
df_test['text_clean'] = df_test['text'].apply(clean_text)

# For this example we use the cleaned text as input.
train_texts = df_train['text_clean']
train_labels = df_train['target']
test_texts = df_test['text_clean']

# ============================================================================
# 5. Set Up the Transformer Model and Tokenizer
# ============================================================================

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# We initialize the model once; note that if you train in CV folds, you may reinitialize or reload weights per fold.
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ============================================================================
# 6. Cross-Validation Training with StratifiedKFold
# ============================================================================

N_FOLDS = 5  # Define the number of folds for cross-validation

oof_preds = np.zeros(len(df_train))
fold_metrics = {}

print(f"Starting {N_FOLDS}-fold cross-validation...")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels), 1):
    print(f"\n===== Fold {fold} =====")
    train_texts_fold = train_texts.iloc[train_idx].reset_index(drop=True)
    train_labels_fold = train_labels.iloc[train_idx].reset_index(drop=True)
    val_texts_fold = train_texts.iloc[val_idx].reset_index(drop=True)
    val_labels_fold = train_labels.iloc[val_idx].reset_index(drop=True)

    # Create datasets
    train_dataset = DisasterDataset(train_texts_fold, train_labels_fold, tokenizer, max_length=128)
    val_dataset = DisasterDataset(val_texts_fold, val_labels_fold, tokenizer, max_length=128)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./results_fold_{fold}',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=42,
        disable_tqdm=False,
        logging_dir=f'./logs_fold_{fold}',
    )

    # Reinitialize the model for each fold for fairness
    model_fold = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    trainer = Trainer(
        model=model_fold,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate fold performance
    eval_results = trainer.evaluate()
    print(f"Fold {fold} evaluation: {eval_results}")
    fold_metrics[fold] = eval_results

    # Get out-of-fold predictions
    preds_output = trainer.predict(val_dataset)
    preds = np.argmax(preds_output.predictions, axis=1)
    oof_preds[val_idx] = preds

# Overall cross-validation performance
from sklearn.metrics import accuracy_score, f1_score
cv_acc = accuracy_score(train_labels, oof_preds)
cv_f1 = f1_score(train_labels, oof_preds)
print("\n===== Overall CV Performance =====")
print(f"Accuracy: {cv_acc:.4f}")
print(f"F1 Score: {cv_f1:.4f}")

# ============================================================================
# 7. Retrain on Full Training Data and Predict on Test Set
# ============================================================================

print("\nRetraining on full training data...")

train_dataset_full = DisasterDataset(train_texts, train_labels, tokenizer, max_length=128)
test_dataset = DisasterDataset(test_texts, labels=None, tokenizer=tokenizer, max_length=128)

training_args_full = TrainingArguments(
    output_dir='./results_full',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    logging_steps=50,
    save_strategy="epoch",
    seed=42,
    logging_dir='./logs_full',
)

# You can start from the best checkpoint from CV or reinitialize
model_full = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

trainer_full = Trainer(
    model=model_full,
    args=training_args_full,
    train_dataset=train_dataset_full,
    compute_metrics=compute_metrics,
)

trainer_full.train()

# Make predictions on the test set
test_preds = trainer_full.predict(test_dataset)
test_pred_labels = np.argmax(test_preds.predictions, axis=1)


