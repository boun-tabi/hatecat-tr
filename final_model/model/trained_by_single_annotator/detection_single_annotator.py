import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from ast import literal_eval
import numpy as np
import evaluate
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from huggingface_hub import login
import wandb
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]
WANDB_TOKEN = os.environ["WANDB_TOKEN"]

def process_dataframe(df, df_not_spans=None, test_ids=None):
    """Helper function to process DataFrames consistently"""
    df['tokens'] = df['tokens'].apply(literal_eval)
    df['tags'] = df['tags'].apply(literal_eval)
    df['Tweet_id'] = df['Tweet_id']
    
    if test_ids is not None:
        df = df[~df['Tweet_id'].isin(test_ids)]
    
    if df_not_spans is not None:
        df = pd.concat([df, df_not_spans])
    
    return df

# Load all DataFrames
base_path = '/users/hasan.seker/baseline'
annotators = ['burak', 'elif', 'irem', 'didem', 'pelin', 'murat']
dfs = {}

# Load full match and non-hateful tweets
df_full = pd.read_csv(f'{base_path}/full_match_no_category.csv')
df_not_spans = pd.read_csv(f'{base_path}/nonhateful_tweets_with_tags.csv')
df_not_spans = process_dataframe(df_not_spans)

# Create test set first
df_full = process_dataframe(df_full)
df_test = df_full.sample(300)
test_ids = df_test['Tweet_id'].tolist()

# Load and process annotator DataFrames
for annotator in annotators:
    file_path = f'{base_path}/annotators/{annotator}/{annotator.capitalize()}_no_category.csv'
    df = pd.read_csv(file_path)
    dfs[annotator] = process_dataframe(df, df_not_spans, test_ids)

# Create datasets dictionary
datasets = {
    annotator: Dataset.from_pandas(df) 
    for annotator, df in dfs.items()
}
datasets['test'] = Dataset.from_pandas(df_test)

# Create DatasetDict
ds = DatasetDict(datasets)

# Set up evaluation
seqeval = evaluate.load("seqeval")
wandb.login(key=WANDB_TOKEN)

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
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        padding=True
    )

    labels = []
    for i in range(len(tokenized_inputs.encodings)):
        label = examples["tags"][i]
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Label mappings
id2label = {
    0: "O",
    1: "B-HATE",
    2: "I-HATE",
}
label2id = {
    "O": 0,
    "B-HATE": 1,
    "I-HATE": 2,
}

label_list = list(id2label.values())
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased", add_prefix_space=True)

# Tokenize datasets
try:
    tokenized_dataset = ds.map(tokenize_and_align_labels, batched=True)
except Exception as e:
    print(f"Error during tokenization: {str(e)}")
    raise

# Training parameters
max_f1 = 0
learn_rate = 5e-5
batch_size = 4
num_epoch = 5
warm_up = 0.1

# Training loop
for dataset in ds.keys():
    if dataset == "test":
        continue
        
    model = AutoModelForTokenClassification.from_pretrained(
        "dbmdz/bert-base-turkish-cased", 
        num_labels=3
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    wandb.init(
        project="detection_annotator",
        name=f"{dataset}_lr-{learn_rate}_bs-{batch_size}_epochs-{num_epoch}_warmup-{warm_up}",
        config={
            "learning_rate": learn_rate,
            "batch_size": batch_size,
            "num_epochs": num_epoch,
            "warm_up": warm_up,
        },
    )

    training_args = TrainingArguments(
        output_dir=f"{dataset}_{learn_rate}_{batch_size}_{num_epoch}_detect",
        learning_rate=learn_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        metric_for_best_model='eval_f1',
        load_best_model_at_end=True,
        greater_is_better=True,
        num_train_epochs=num_epoch,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=1,
        push_to_hub=False,
        warmup_ratio=warm_up,
        report_to="wandb",
    )

    # Split dataset
    split_ds = tokenized_dataset[dataset].train_test_split(test_size=0.1, seed=42)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_ds["train"],
        eval_dataset=split_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"Evaluation results for {dataset}:", eval_results)

    # Random sample prediction
    ix = np.random.randint(len(tokenized_dataset["test"]))
    predictions, labels, metrics = trainer.predict(tokenized_dataset["test"])
    
    print(f"Test set metrics for {dataset}:", metrics)
    print('Text:', tokenized_dataset["test"]['Text'][ix])
    print('Label:', ''.join([
        tokenizer.convert_ids_to_tokens(token) if label in list(range(1,4)) else '-' 
        for label, token in zip(tokenized_dataset["test"]['labels'][ix], tokenized_dataset["test"]['input_ids'][ix])
    ]))

    # Push to hub and cleanup
    login(HF_TOKEN)
    trainer.push_to_hub()
    wandb.finish()