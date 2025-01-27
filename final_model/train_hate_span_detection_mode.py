import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from ast import literal_eval
import numpy as np
import evaluate
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from huggingface_hub import login
from tabulate import tabulate
from dotenv import load_dotenv
import os 

load_dotenv()
HF_TOKEN =  os.environ["HF_TOKEN"]
df_spans = pd.read_csv(f'/users/hasan.seker/baseline/full_match_no_category.csv')
df_not_spans = pd.read_csv(f'/users/hasan.seker/baseline/nonhateful_tweets_with_tags.csv')

#df_all = pd.concat([df_spans, df_not_spans.sample(df_spans.shape[0])])
#df_all = pd.concat([df_spans.sample(1000), df_not_spans])
df_all = pd.concat([df_spans, df_not_spans])


df_all['tokens'] = df_all['tokens'].apply(literal_eval)
df_all['tags'] = df_all['tags'].apply(literal_eval)

train_df, temp_df = train_test_split(df_all, test_size=0.2, random_state=42)

# Split temp set into validation and test sets (50% validation, 50% test of the temp set which is 10% each of the original)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Convert to Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Create DatasetDict
ds = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

seqeval = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)


    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l]  for (p, l) in zip(prediction, label) if l != -100]
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
    for i, label in enumerate(examples[f"tags"]):

        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            else: # label all tokens of a given word.
              label_ids.append(label[word_idx])
            # Only label the first token of a given word.
            # elif word_idx != previous_word_idx:
            #    label_ids.append(label[word_idx])
            # else:
            #    label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

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
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased",add_prefix_space=True)
tokenized_dataset = ds.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased", num_labels=3
)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

experiments = []
best_experiment = {}
best_model = None
max_f1 = 0
for _ in range(1):
    training_args = TrainingArguments(
        output_dir='bert-base-turkish-cased_hate_span_detection_final/',
        #learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        greater_is_better=False,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=1,
        push_to_hub=False,
        #weight_decay=0.01,
        #report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.evaluate(tokenized_dataset["test"])
    predictions, label_ids, metrics = trainer.predict(tokenized_dataset["test"])
    experiments.append({'metrics': metrics})
    print(metrics)
    if metrics['test_f1'] > max_f1:
        max_f1 = metrics['test_f1']
        best_experiment = {'metrics': metrics}
        best_model = trainer

ix = np.random.randint(len(tokenized_dataset["test"]))

predictions, labels, metrics = best_model.predict(tokenized_dataset["test"])
print("Test set metrics:", metrics)

# Same for validation set
val_predictions, val_labels, val_metrics = best_model.predict(tokenized_dataset["validation"])
print("Validation set metrics:", val_metrics)


print('Text', tokenized_dataset["test"]['Text'][ix])
print('Label', ''.join([tokenizer.convert_ids_to_tokens(token) if label in list(range(1,4)) else '-' for label, token in zip(tokenized_dataset["test"]['labels'][ix],tokenized_dataset["test"]['input_ids'][ix] ) ]))
' '.join([tokenizer.convert_ids_to_tokens(token) if label in list(range(1,4)) else '-' for label, token in zip( np.argmax(predictions[ix],axis=1)[:len(tokenized_dataset["test"]['input_ids'][ix])],tokenized_dataset["test"]['input_ids'][ix] ) ]).replace(' ##' , '')
login(HF_TOKEN)

for i in experiments:
    print(i)

print("Best experiment", best_experiment)

best_model.push_to_hub()

# Function to decode tokens and labels
def decode_tokens_and_labels(token_ids, label_ids, predictions):
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    decoded_labels = [id2label[label] if label != -100 else "PAD" for label in label_ids]
    decoded_predictions = [id2label[np.argmax(pred)] if pred is not None else "PAD" for pred in predictions]
    return decoded_tokens, decoded_labels, decoded_predictions

# Find correct and incorrect predictions
correct = []
incorrect = []

for idx in range(len(tokenized_dataset["test"])):
    token_ids = tokenized_dataset["test"]["input_ids"][idx]
    label_ids = tokenized_dataset["test"]["labels"][idx]
    preds = predictions[idx]

    # Decode tokens, labels, and predictions
    tokens, labels, preds_decoded = decode_tokens_and_labels(token_ids, label_ids, preds)
    
    # Remove padding and focus on valid tokens
    valid_tokens = [t for t, l in zip(tokens, labels) if l != "PAD"]
    valid_labels = [l for l in labels if l != "PAD"]
    valid_preds = [p for p, l in zip(preds_decoded, labels) if l != "PAD"]

    if valid_labels == valid_preds:
        correct.append({"tokens": valid_tokens, "labels": valid_labels, "predictions": valid_preds})
    else:
        incorrect.append({"tokens": valid_tokens, "labels": valid_labels, "predictions": valid_preds})

# Print 5 correct predictions in a tabular format
print("\n5 Correct Predictions:")
for example in correct[:3]:
    table_data = []
    for token, label, prediction in zip(example['tokens'], example['labels'], example['predictions']):
        table_data.append([token, label, prediction])
    
    print(tabulate(table_data, headers=["Token", "True Label", "Prediction"], tablefmt="grid"))
    print("-" * 50)

# Print 5 incorrect predictions in a tabular format
print("\n5 Incorrect Predictions:")
for example in incorrect[:15]:
    table_data = []
    for token, label, prediction in zip(example['tokens'], example['labels'], example['predictions']):
        table_data.append([token, label, prediction])
    
    print(tabulate(table_data, headers=["Token", "True Label", "Prediction"], tablefmt="grid"))
    print("-" * 50)
