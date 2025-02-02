import pandas as pd
from sklearn.model_selection import KFold
from datasets import Dataset, DatasetDict
from ast import literal_eval
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from huggingface_hub import login
import wandb
from dotenv import load_dotenv
import os
from collections import defaultdict

# Load environment variables
load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]
WANDB_TOKEN = os.environ["WANDB_TOKEN"]

# Load and preprocess data
df_spans = pd.read_csv('/users/hasan.seker/baseline/all_annotations_no_category.csv')
df_not_spans = pd.read_csv('/users/hasan.seker/baseline/nonhateful_tweets_with_tags.csv')
df_all = pd.concat([df_spans, df_not_spans])

df_all['tokens'] = df_all['tokens'].apply(literal_eval)
df_all['tags'] = df_all['tags'].apply(literal_eval)

# Initialize evaluation metric
seqeval = evaluate.load("seqeval")
wandb.login(key=WANDB_TOKEN)

# Label mappings for binary classification
id2label = {
    0: "O",
    1: "B-HATE",
    2: "I-HATE",
}
label2id = {v: k for k, v in id2label.items()}
label_list = list(id2label.values())

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    print(results)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased", add_prefix_space=True)

# Hyperparameter grid
param_grid = {
    #'learning_rate': [1e-6, 1e-5, 5e-5],
    'learning_rate': [5e-5],
    #'batch_size': [16, 32, 64],
    'batch_size': [32, 64],
    'num_epochs': [5, 10],
    'warm_up': [0, 0.1]
}

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store results for each fold
fold_results = defaultdict(list)
best_overall = {
    'f1': 0,
    'config': None,
    'fold_metrics': None
}

# Iterate through hyperparameter combinations
for lr in param_grid['learning_rate']:
    for bs in param_grid['batch_size']:
        for epochs in param_grid['num_epochs']:
            for warm_up in param_grid['warm_up']:
                print(f"\nTesting: lr={lr}, bs={bs}, epochs={epochs}, warm_up={warm_up}")
                
                fold_metrics = []
                for fold, (train_idx, val_idx) in enumerate(kf.split(df_all)):
                    print(f"\nFold {fold + 1}/5")
                    
                    # Split data for this fold
                    train_fold = df_all.iloc[train_idx]
                    val_fold = df_all.iloc[val_idx]
                    
                    # Convert to datasets
                    train_dataset = Dataset.from_pandas(train_fold)
                    val_dataset = Dataset.from_pandas(val_fold)
                    
                    # Create dataset dictionary
                    ds = DatasetDict({
                        'train': train_dataset,
                        'validation': val_dataset
                    })
                    
                    # Tokenize datasets
                    tokenized_dataset = ds.map(tokenize_and_align_labels, batched=True)
                    
                    # Initialize model for binary classification
                    model = AutoModelForTokenClassification.from_pretrained(
                        "dbmdz/bert-base-turkish-cased",
                        num_labels=3  # O, B-HATE, I-HATE
                    )
                    
                    # Initialize data collator
                    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
                    
                    # Initialize wandb run
                    run_name = f"fold{fold+1}_lr{lr}_bs{bs}_epochs{epochs}_warmup{warm_up}"
                    wandb.init(
                        project="detection",  # Changed to match your project name
                        name=run_name,
                        config={
                            "learning_rate": lr,
                            "batch_size": bs,
                            "num_epochs": epochs,
                            "warm_up": warm_up,
                            "fold": fold + 1
                        }
                    )
                    
                    # Training arguments
                    training_args = TrainingArguments(
                        output_dir=f"fold{fold+1}_{lr}_{bs}_{epochs}_detect",  # Changed to match your naming
                        learning_rate=lr,
                        per_device_train_batch_size=bs,
                        per_device_eval_batch_size=bs,
                        num_train_epochs=epochs,
                        evaluation_strategy="epoch",
                        save_strategy="epoch",
                        metric_for_best_model='eval_loss',
                        load_best_model_at_end=True,
                        greater_is_better=False,
                        logging_dir="./logs",
                        logging_steps=100,
                        save_total_limit=1,
                        push_to_hub=False,
                        report_to="wandb",
                        warmup_ratio=warm_up
                    )
                    
                    # Initialize trainer
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_dataset["train"],
                        eval_dataset=tokenized_dataset["validation"],
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics,
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
                    )
                    
                    # Train and evaluate
                    trainer.train()
                    metrics = trainer.evaluate()
                    fold_metrics.append(metrics)
                    
                    wandb.finish()
                
                # Calculate mean and std of metrics across folds
                mean_metrics = {
                    'precision': np.mean([m['eval_precision'] for m in fold_metrics]),
                    'recall': np.mean([m['eval_recall'] for m in fold_metrics]),
                    'f1': np.mean([m['eval_f1'] for m in fold_metrics]),
                    'accuracy': np.mean([m['eval_accuracy'] for m in fold_metrics])
                }
                
                std_metrics = {
                    'precision': np.std([m['eval_precision'] for m in fold_metrics]),
                    'recall': np.std([m['eval_recall'] for m in fold_metrics]),
                    'f1': np.std([m['eval_f1'] for m in fold_metrics]),
                    'accuracy': np.std([m['eval_accuracy'] for m in fold_metrics])
                }
                
                # Store results
                config_key = f"lr={lr}_bs={bs}_epochs={epochs}_warmup={warm_up}"
                fold_results[config_key] = {
                    'mean': mean_metrics,
                    'std': std_metrics
                }
                
                # Update best overall if necessary
                if mean_metrics['f1'] > best_overall['f1']:
                    best_overall['f1'] = mean_metrics['f1']
                    best_overall['config'] = {
                        'learning_rate': lr,
                        'batch_size': bs,
                        'num_epochs': epochs,
                        'warm_up': warm_up
                    }
                    best_overall['fold_metrics'] = fold_metrics

# Print results
print("\nResults for all configurations:")
for config, results in fold_results.items():
    print(f"\n{config}")
    print("Mean metrics:")
    for metric, value in results['mean'].items():
        print(f"{metric}: {value:.4f} ± {results['std'][metric]:.4f}")

print("\nBest configuration:")
print(f"Learning rate: {best_overall['config']['learning_rate']}")
print(f"Batch size: {best_overall['config']['batch_size']}")
print(f"Number of epochs: {best_overall['config']['num_epochs']}")
print(f"Warm up ratio: {best_overall['config']['warm_up']}")
print(f"Best F1 score: {best_overall['f1']:.4f}")