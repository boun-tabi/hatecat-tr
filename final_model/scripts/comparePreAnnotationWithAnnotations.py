import json
import ast

def load_json_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip()
            # Convert JSON literals to Python literals
            line = line.replace("null", "None")
            line = line.replace("false", "False")
            line = line.replace("true", "True")
            try:
                obj = ast.literal_eval(line)
                data.append(obj)
            except Exception as e:
                print(f"Error parsing line: {line[:80]}... -> {e}")
        return data

def build_id_to_annotations_file2(file2_data):
    id_to_ann = {}
    for entry in file2_data:
        tweet_id = entry.get('data', {}).get('id')
        if tweet_id:
            # Since file2 is now gpt_annotations (original file1 format)
            id_to_ann[tweet_id] = entry.get('predictions') or entry.get('annotations')
    return id_to_ann

def extract_comparable_result(annotation):
    if annotation and isinstance(annotation, list) and len(annotation) > 0:
        first_item = annotation[0]
        # Check if first item is a dictionary
        if isinstance(first_item, dict):
            if 'prediction' in first_item:
                return first_item['prediction'].get('result', [])
            if 'result' in first_item:
                return first_item['result']
        # If first item is not a dict, return the annotation as is
        # This handles cases where annotation is already a list of results
        return annotation
    return []

# write_once = False
def normalize_item(item):
    val = item.get('value', {})
    return (
        val.get('start'),
        val.get('end'),
        val.get('text', '').strip(),
        tuple(sorted(val.get('labels', [])))
    )

def compare_annotations(ann1, ann2):
    global write_once
    result1 = extract_comparable_result(ann1)
    result2 = extract_comparable_result(ann2)

    norm1_set = set(normalize_item(x) for x in result1)
    norm2_set = set(normalize_item(x) for x in result2)

    # if not write_once:
    #     print("------")
    #     print("Normalized1 SET:", norm1_set)
    #     print("Normalized2 SET:", norm2_set)
    #     print(norm1_set != norm2_set)
    #     write_once = True

    return norm1_set != norm2_set

def main():
    # Load data from your txt files - SWAPPED the file paths
    file1_data = load_json_file('../merged_annotations/all_annotations_tr.txt')
    file2_data = load_json_file('../gpt_predictions/gpt_tr_anno.txt')

    file1_ids = set(item.get('data', {}).get('id') for item in file1_data if item.get('data', {}).get('id'))
    file2_ids = set(entry.get('data', {}).get('id') for entry in file2_data if entry.get('data', {}).get('id'))

    # print(f"Number of tweets in file1: {len(file1_data)}")
    # print(f"Number of tweets in file2: {len(file2_data)}")

    
    # Build quick lookup from file2
    file2_lookup = build_id_to_annotations_file2(file2_data)

    dropped_count = 0
    differing_count = 0
    same_count = 0

    accepted_examples = []
    modified_examples = []

    for item in file1_data:
        tweet_id = item.get('data', {}).get('id')
        # Since file1 is now all_annotations_merged (original file2 format)
        ann1 = item.get('annotations') or item.get('predictions')

        if not tweet_id:
            continue  # skip if no ID

        if tweet_id not in file2_lookup:
            dropped_count += 1
        else:
            ann2 = file2_lookup[tweet_id]
            if compare_annotations(ann1, ann2):
                differing_count += 1
                if len(modified_examples) < 5:
                    modified_examples.append({
                        "id": tweet_id,
                        "file1": ann1,
                        "file2": ann2
                    })
            else:
                same_count += 1
                if len(accepted_examples) < 5:
                    accepted_examples.append({
                        "id": tweet_id,
                        "file1": ann1,
                        "file2": ann2
                    })

    print("\nSummary:")
    print(f"Total tweets: {len(file1_data)}")
    print(f"Number of dropped tweets: {len(file2_data)-len(file1_data)}")
    print(f"Number of differing annotations: {differing_count}")
    print(f"Number of tweets accepted as it is: {same_count}")

    # Print examples
    # print("\nExamples of ACCEPTED tweets (matched and same annotations):")
    # for ex in accepted_examples:
    #     print(f"\nTweet ID: {ex['id']}")
    #     print("File1 annotations:", extract_comparable_result(ex['file1']))
    #     print("File2 annotations:", extract_comparable_result(ex['file2']))

    # print("\nExamples of MODIFIED tweets (matched but different annotations):")
    # for ex in modified_examples:
    #     print(f"\nTweet ID: {ex['id']}")
    #     print("File1 annotations:", extract_comparable_result(ex['file1']))
    #     print("File2 annotations:", extract_comparable_result(ex['file2']))

if __name__ == "__main__":
    main()