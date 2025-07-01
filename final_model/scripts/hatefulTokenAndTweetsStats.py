import numpy as np
import re 
import ast

def labelStudioTokenize(text):
    return re.findall(r'\b\w+\b', text)



CATEGORY_MAPPING = {
    # Turkish to English
    'Abartma, Genelleme, Yükleme, Çarpıtma': 'Exaggeration; Generalization; Attribution; Distortion',
    'Düşmanlık, Savaş, Saldırı, Öldürme, Yaralama Tehditi': 'Threat of Enmity; War; Attack; Murder; or Harm',
    'Küfür, Hakaret, Aşağılama, İnsandışılaştırma': 'Swearing; Insult; Defamation; Dehumanization',
    'Dışlama; Ayrımcı Söylem': 'Exclusive/Discriminatory Discourse',
    'Simgeleştirme': 'Symbolization',
}

def normalize_category(category):
    return CATEGORY_MAPPING.get(category, category)

def anno_stats(filename):
    total_span_count = 0
    category_tweet_count = {}
    category_token_count = {}
    length_list = []
    tweets = []
    if filename == "all_annotations_combined.txt":
        print("Stats for Arabic and Turkish Tweets Combined")
    elif filename == "all_annotations_tr.txt":
        print("Stats for Turkish Tweets")
    elif filename == "all_annotations_ar.txt":
        print("Stats for Arabic Tweets")
    filename = "../merged_annotations/"+filename
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            json_line = ast.literal_eval(line.strip())
            annos = json_line['annotations'][0]['result']
            tweet_categories = set()
            start_end_indices = set()
            for anno in annos:
                span = anno['value']['text']
                tokens = span
                start = anno['value']['start']
                end = anno['value']['end']
                length = len(anno['value']['text'])
                length_list.append(length)
                tweets.append((length, span))
                number_of_tokens = 1
                category = normalize_category(anno['value']['labels'][0])
                if not(category in tweet_categories):
                    tweet_categories.add(category)
                    category_tweet_count[category] = category_tweet_count.get(category, 0) + 1
                if not((start, end) in start_end_indices):
                    start_end_indices.add((start, end))
                    total_span_count += 1
                category_token_count[category] = category_token_count.get(category, 0) + number_of_tokens
        print(f"Number of tweets: {len(lines)}")
        print(f"Total number of spans: {total_span_count}")
        print(f"Average span per tweet: {total_span_count / len(lines) if lines else 0:.2f}")
    # Sort tweets by length in descending order and get the longest ten tweets
    longest_tweets = sorted(tweets, key=lambda x: x[0], reverse=True)[:10]

    # Print the longest ten tweets
    print("\nLongest Ten Tweets:")
    for length, tweet in longest_tweets:
        print(f"Length: {length}, Tweet: {tweet}")

    mean = sum(length_list) / len(length_list) if length_list else 0
    stdev = np.std(length_list)
    print(f"\nSpan Length Statistics:")
    print(f"Mean: {mean:.2f}")
    print(f"Standard Deviation: {stdev:.2f}")
    for category, count in category_token_count.items():
        print(f"{category}: {count} spans")
    print("\n", "-"*50, "\n")
    for category, count in category_tweet_count.items():
        print(f"{category}: {count} tweets")

anno_stats("all_annotations_tr.txt")
anno_stats("all_annotations_ar.txt")
anno_stats("all_annotations_combined.txt")
