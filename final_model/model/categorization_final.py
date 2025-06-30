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
import wandb
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN =  os.environ["HF_TOKEN"]
WANDB_TOKEN = os.environ["WANDB_TOKEN"]

df_all_anno = pd.read_csv(f'../merged_annotations/all_annotations_tr.csv')
df_gpt = pd.read_csv(f'../gpt_predictions/gpt_anno.csv')
df_full = pd.read_csv(f'../matching_annotations/full_match.csv')

df_not_spans = pd.read_csv(f'../non_hateful/nonhateful_tweets_with_tags.csv')

df_full['tokens'] = df_full['tokens'].apply(literal_eval)
df_full['tags'] = df_full['tags'].apply(literal_eval)
df_full['Tweet_id'] = df_full['Tweet_id']

df_gpt['tokens'] = df_gpt['tokens'].apply(literal_eval)
df_gpt['tags'] = df_gpt['tags'].apply(literal_eval)
df_gpt['Tweet_id'] = df_gpt['Tweet_id']

df_all_anno['tokens'] = df_all_anno['tokens'].apply(literal_eval)
df_all_anno['tags'] = df_all_anno['tags'].apply(literal_eval)
df_all_anno['Tweet_id'] = df_all_anno['Tweet_id']

df_not_spans['tokens'] = df_not_spans['tokens'].apply(literal_eval)
df_not_spans['tags'] = df_not_spans['tags'].apply(literal_eval)
df_not_spans['Tweet_id'] = df_not_spans['Tweet_id']

#df_all = pd.concat([df_spans, df_not_spans.sample(df_spans.shape[0])])
df_all_gpt = pd.concat([df_gpt, df_not_spans])
df_all_full = pd.concat([df_full, df_not_spans])
df_all_anno2 = pd.concat([df_all_anno, df_not_spans])

df_test = df_full.sample(300)

df_all_anno2 = df_all_anno2[~df_all_anno2['Tweet_id'].isin(df_test['Tweet_id'].tolist())]
df_all_gpt = df_all_gpt[~df_all_gpt['Tweet_id'].isin(df_test['Tweet_id'].tolist())]
df_all_full = df_all_full[~df_all_full['Tweet_id'].isin(df_test['Tweet_id'].tolist())]



# Convert to Dataset
all_dataset = Dataset.from_pandas(df_all_anno2)
gpt_dataset = Dataset.from_pandas(df_all_gpt)
full_dataset = Dataset.from_pandas(df_all_full)
test_dataset = Dataset.from_pandas(df_test)

# Create DatasetDict
ds = DatasetDict({
    'all': all_dataset,
    'gpt': gpt_dataset,
    'full': full_dataset,
    'test':test_dataset 
})

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
    1: "I-Exclusive/Discriminatory Discourse",
    2: "I-Exaggeration; Generalization; Attribution; Distortion",
    3: "I-Threat of Enmity; War; Attack; Murder; or Harm",
    4: "I-Symbolization",
    5: "I-Swearing; Insult; Defamation; Dehumanization",
}

label2id = {
    "O": 0,
    "I-Exclusive/Discriminatory Discourse": 1,
    "I-Exaggeration; Generalization; Attribution; Distortion": 2,
    "I-Threat of Enmity; War; Attack; Murder; or Harm": 3,
    "I-Symbolization": 4,
    "I-Swearing; Insult; Defamation; Dehumanization": 5,
}
label_list = list(id2label.values())
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased",add_prefix_space=True)
tokenized_dataset = ds.map(tokenize_and_align_labels, batched=True)
max_f1 = 0
learn_rate = 5e-5
batch_size = 4
num_epoch = 10
warm_up = 0.1
for dataset in ds.keys():
    if dataset == "test":
        continue
    model = AutoModelForTokenClassification.from_pretrained(
        "dbmdz/bert-base-turkish-cased", num_labels=6
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    wandb.init(
        project="categorization_final",  # Replace with your project name
        name=f"{dataset}_lr-{learn_rate}_bs-{batch_size}_epochs-{num_epoch}_warmup-{warm_up}",  # Unique name
        config={
            "learning_rate": learn_rate,
            "batch_size": batch_size,
            "num_epochs": num_epoch,
            "warm_up": warm_up,
        },
    )
    training_args = TrainingArguments(
        output_dir=f"{dataset}_{learn_rate}_{batch_size}_{num_epoch}_categorize",
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
        #weight_decay=0.01,
        report_to="wandb",
    )
    split_ds = tokenized_dataset[dataset].train_test_split(test_size=0.1, seed=42)
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
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.evaluate(tokenized_dataset["test"])
    predictions, label_ids, metrics = trainer.predict(tokenized_dataset["test"])
    print(metrics)
    

    ix = np.random.randint(len(tokenized_dataset["test"]))

    predictions, labels, metrics = trainer.predict(tokenized_dataset["test"])
    print("Test set metrics:", metrics)

    print('Text', tokenized_dataset["test"]['Text'][ix])
    print('Label', ''.join([tokenizer.convert_ids_to_tokens(token) if label in list(range(1,6)) else '-' for label, token in zip(tokenized_dataset["test"]['labels'][ix],tokenized_dataset["test"]['input_ids'][ix] ) ]))
    ' '.join([tokenizer.convert_ids_to_tokens(token) if label in list(range(1,6)) else '-' for label, token in zip( np.argmax(predictions[ix],axis=1)[:len(tokenized_dataset["test"]['input_ids'][ix])],tokenized_dataset["test"]['input_ids'][ix] ) ]).replace(' ##' , '')
    login(HF_TOKEN)

    print("Result:", metrics, dataset)

    trainer.push_to_hub()
    wandb.finish()

