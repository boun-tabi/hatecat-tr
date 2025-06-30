import json
import csv
import ast

import re

# Open and read the file
input_file = "../merged_annotations/all_annotations_tr.txt"  # Replace with your file name
output_file = "../merged_annotations/all_annotations_tr.csv"
output_file_without_categories = "../merged_annotations/all_annotations_tr_no_category.csv"
mode = "w"

# Mappings for categories to their B/I tags
label2id = {
    "Exclusive/Discriminatory Discourse": (1),
    "Exaggeration; Generalization; Attribution; Distortion": (2),
    "Threat of Enmity; War; Attack; Murder; or Harm": (3),
    "Symbolization": (4),
    "Swearing; Insult; Defamation; Dehumanization": (5),
}

tr_to_en_categories_mapping = {
    "Dışlama; Ayrımcı Söylem": "Exclusive/Discriminatory Discourse",
    "Düşmanlık, Savaş, Saldırı, Öldürme, Yaralama Tehditi": "Threat of Enmity; War; Attack; Murder; or Harm",
    "Simgeleştirme": "Symbolization",
    "Abartma, Genelleme, Yükleme, Çarpıtma": "Exaggeration; Generalization; Attribution; Distortion",
    "Küfür, Hakaret, Aşağılama, İnsandışılaştırma": "Swearing; Insult; Defamation; Dehumanization",
}
# Function to tokenize text and generate tags
def generate_tokens_and_tags(text, span_start=None, span_end=None):
    tokens = text.split()
    tags = [0] * len(tokens)  # Default all tags to 0 (outside)
    
    if span_start is not None and span_end is not None:
        current_pos = 0
        for i, token in enumerate(tokens):
            token_start = current_pos
            token_end = current_pos + len(token)
            if token_start >= span_start and token_end <= span_end:
                tags[i] = 2 if token_start > span_start else 1  # 2 for inside, 1 for beginning
            current_pos = token_end + 1  # Account for space
    return tokens, tags

# Function to tokenize text
def tokenize(text):
    return re.findall(r'\b\w+\b', text)

multiclass_tokens = 0
multiclass_tweets = 0

def annotate_tweet(tweet, spans_with_labels):
    global multiclass_tokens, multiclass_tweets
    no_multiclass = True
    words = tokenize(tweet)
    labels = [set({0}) for _ in range(len(words))]  # Initialize all words with empty sets
    labels_without_classes = [0 for _ in range(len(words))]
    def find_span_start(words, span):
        span_words = tokenize(span)
        for i in range(len(words) - len(span_words) + 1):
            if words[i:i+len(span_words)] == span_words:
                return i
        return -1

    for span, category in spans_with_labels:
        start_index = find_span_start(words, span)
        if category not in label2id:
            category = tr_to_en_categories_mapping.get(category, category)
        if start_index != -1 and category in label2id:
            tag = label2id[category]
            labels_without_classes[start_index] = 1
            labels[start_index].add(tag)
            if len(labels[start_index]) > 2:  # If already assigned. all sets have 0 tag so it must be bigger than 2
                multiclass_tokens += 1
                no_multiclass = False
                print(labels[start_index])
                print(tweet)
                print(tokenize(tweet)[start_index])
            for i in range(start_index+1, start_index + len(tokenize(span))):
                labels[i].add(tag)
                labels_without_classes[i] = 2
                if len(labels[i]) > 2:  # If already assigned. all sets have 0 tag so it must be bigger than 2
                    multiclass_tokens += 1
                    no_multiclass = False
                    print(labels[i])
                    print(tweet)
                    print(tokenize(tweet)[i])
                    
    if not no_multiclass:
        multiclass_tweets += 1
        #print(f"Multiclass tweet: {tweet}")
    return words, [max(lbl) for lbl in labels], labels_without_classes
    #return words, [list(lbl) for lbl in labels]  # Convert sets to lists for JSON serialization




with open(input_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

data = []
for line in lines:
    item = ast.literal_eval(line.strip())  # Parse line to dictionary
    tweet_id = item["data"]["id"]
    target = item["data"]["target"]
    text = item["data"]["text"]
    annotations = item["annotations"][0]["result"]
    
    spans_with_labels = []
    if annotations:  # If annotations exist
        for annotation in annotations:
            span_start = annotation["value"]["start"]
            span_end = annotation["value"]["end"]
            span = annotation["value"]["text"]
            label = annotation["value"]["labels"][0] 
            spans_with_labels.append((span, label))
    else:  # No annotations
        spans_with_labels = []

    tokens, tags, tags_without_classes = annotate_tweet(text, spans_with_labels)
    data.append([tweet_id, target, text, ",".join([s[0] for s in spans_with_labels]), tokens, tags, tags_without_classes])


# Write to CSV
with open(output_file, mode, encoding="utf-8", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Tweet_id", "Target", "Text", "Span", "tokens", "tags"])
    for row in data:
        csvwriter.writerow([row[0], row[1], row[2], row[3], json.dumps(row[4]), json.dumps(row[5])])

with open(output_file_without_categories, mode, encoding="utf-8", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Tweet_id", "Target", "Text", "Span", "tokens", "tags"])
    for row in data:
        csvwriter.writerow([row[0], row[1], row[2], row[3], json.dumps(row[4]), json.dumps(row[6])])

print(f"Data transformed and saved to {output_file}")
print(f"Multiclass tokens: {multiclass_tokens}")
print(f"Multiclass tweets: {multiclass_tweets}")