#!/usr/bin/env python3
import argparse
import json
import os
from ast import literal_eval
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


CATEGORIZATION_ID2LABEL = {
    0: "O",
    1: "I-Exclusive/Discriminatory Discourse",
    2: "I-Exaggeration; Generalization; Attribution; Distortion",
    3: "I-Threat of Enmity; War; Attack; Murder; or Harm",
    4: "I-Symbolization",
    5: "I-Swearing; Insult; Defamation; Dehumanization",
}
CATEGORIZATION_LABEL2ID = {v: k for k, v in CATEGORIZATION_ID2LABEL.items()}

DETECTION_ID2LABEL = {
    0: "O",
    1: "B-HATE",
    2: "I-HATE",
}
DETECTION_LABEL2ID = {v: k for k, v in DETECTION_ID2LABEL.items()}

def load_dataframes(
    train_set_path: str,
    test_set_path: str,
    non_hateful_path: str,
    test_sample_size: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_train_set_merged_with_non_hateful, df_test)
    """
    df_train_set = pd.read_csv(train_set_path)
    df_test_set = pd.read_csv(test_set_path)
    df_not_spans = pd.read_csv(non_hateful_path)

    for df in (df_test_set, df_train_set, df_not_spans):
        df["tokens"] = df["tokens"].apply(literal_eval)
        df["tags"] = df["tags"].apply(literal_eval)
        df["Tweet_id"] = df["Tweet_id"]

    df_train_set_merged_with_non_hateful = pd.concat([df_train_set, df_not_spans], ignore_index=True)

    if test_sample_size > 0:
        df_test = df_test_set.sample(test_sample_size, random_state=random_state)
    else:
        df_test = df_test_set.copy()

    exclude_ids = set(df_test["Tweet_id"].tolist())
    df_train_set_merged_with_non_hateful = df_train_set_merged_with_non_hateful[~df_train_set_merged_with_non_hateful["Tweet_id"].isin(exclude_ids)]

    return df_train_set_merged_with_non_hateful, df_test


def make_hf_datasets(df_train_set_merged_with_non_hateful: pd.DataFrame, df_test: pd.DataFrame) -> DatasetDict:
    train_dataset = Dataset.from_pandas(df_train_set_merged_with_non_hateful)
    test_dataset = Dataset.from_pandas(df_test)

    ds = DatasetDict({"train": train_dataset, "test": test_dataset})
    return ds

def build_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    return tokenizer


def align_labels_with_tokens(examples, tokenizer, label_list: List[str]):
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


def build_metrics(id2label: Dict[int, str]):
    seqeval = evaluate.load("seqeval")
    label_list = list(id2label.values())

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p_i] for (p_i, l_i) in zip(prediction, label) if l_i != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l_i] for (p_i, l_i) in zip(prediction, label) if l_i != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results.get("overall_precision", 0.0),
            "recall": results.get("overall_recall", 0.0),
            "f1": results.get("overall_f1", 0.0),
            "accuracy": results.get("overall_accuracy", 0.0),
        }

    return compute_metrics

def build_trainer(
    tokenized_dataset: DatasetDict,
    model_name: str,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    warmup_ratio: float,
    weight_decay: float,
    early_stop_patience: int,
    logging_steps: int = 100,
    eval_train_split_ratio: float = 0.1,
    seed: int = 42,
):
    num_labels = len(id2label)

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)
    tokenizer = build_tokenizer(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = build_metrics(id2label)

    output_dir = f"train"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        metric_for_best_model="eval_f1",
        load_best_model_at_end=True,
        greater_is_better=True,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=logging_steps,
        save_total_limit=1,
        push_to_hub=False,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
    )

    split_ds = tokenized_dataset["train"].train_test_split(test_size=eval_train_split_ratio, seed=seed)
    train_dataset = split_ds["train"]
    eval_dataset = split_ds["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stop_patience)],
    )

    return trainer, tokenizer

def parse_args():
    p = argparse.ArgumentParser(description="Hate Speech Categorization or Detection Model Trainer")

    # Task / Labels
    p.add_argument("--task", type=str, choices=["categorization", "detection"], required=True,
                   help="Choose label schema: 'categorization' (detects and categorizes the hate speech containing spans) or 'detection' (detects the hate speech containing spans).")

    # Data
    p.add_argument("--train_set", type=str, required=True,
                   help="CSV with columns: Tweet_id, tokens(list), tags(list) for training.")
    p.add_argument("--test_set", type=str, required=True,
                   help="CSV with columns: Tweet_id, tokens(list), tags(list) to sample for the test set.")
    p.add_argument("--non_hateful", type=str, required=True,
                   help="CSV with columns: Tweet_id, tokens(list), tags(list) for non-hateful tweets to concatenate to training set.")

    # Model / Training
    p.add_argument("--model_name", type=str, default="dbmdz/bert-base-turkish-cased", help="HF model checkpoint.")
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--eval_train_split_ratio", type=float, default=0.1, help="Fraction of training data to use for evaluation.")
    p.add_argument("--test_sample_size", type=int, default=300, help="Number of examples to sample from 'test_set' as held-out test.")
    p.add_argument("--random_state", type=int, default=42, help="Random seed for sampling.")

    # Misc
    p.add_argument("--log_steps", type=int, default=100)

    return p.parse_args()


def main():
    args = parse_args()

    if args.task == "categorization":
        id2label = CATEGORIZATION_ID2LABEL
        label2id = CATEGORIZATION_LABEL2ID
    else:
        id2label = DETECTION_ID2LABEL
        label2id = DETECTION_LABEL2ID

    df_train_set_merged_with_non_hateful, df_test = load_dataframes(
        args.train_set,
        args.test_set,
        args.non_hateful,
        args.test_sample_size,
        args.random_state,
    )

    ds = make_hf_datasets(df_train_set_merged_with_non_hateful, df_test)

    tokenizer = build_tokenizer(args.model_name)

    def _map_fn(examples):
        return align_labels_with_tokens(examples, tokenizer, list(id2label.values()))

    tokenized_dataset = ds.map(_map_fn, batched=True)

    trainer, tok = build_trainer(
        tokenized_dataset=tokenized_dataset,
        model_name=args.model_name,
        id2label=id2label,
        label2id=label2id,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        logging_steps=args.log_steps,
        eval_train_split_ratio=args.eval_train_split_ratio,
        seed = args.random_state
    )

    trainer.train()
    trainer.evaluate(tokenized_dataset["test"])

    predictions, label_ids, metrics = trainer.predict(tokenized_dataset["test"])
    print(f"[{args.task.upper()}] Test metrics:", metrics)

if __name__ == "__main__":
    main()