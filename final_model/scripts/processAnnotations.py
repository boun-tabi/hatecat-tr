import json
import os
from collections import defaultdict
import numpy as np
import csv
import uuid
import re

iou_threshold_for_full_match = 0.65

didem = open("../mismatching_annotations/turkish/didem.txt", "w")
murat = open("../mismatching_annotations/turkish/murat.txt", "w")
burak = open("../mismatching_annotations/turkish/burak.txt", "w")
irem = open("../mismatching_annotations/turkish/irem.txt", "w")
arabic = open("../mismatching_annotations/arabic/arabic.txt", "w")

assignment_list = ["Didem","Murat","Burak","İrem"]
assignment_count = [0,0,0,0]
assignment_files = [didem,murat,burak,irem,arabic]

full_match_file_tr = open("../matching_annotations/full_match.txt", "w")
full_match_file_ar = open("../matching_annotations/full_match_ar.txt", "w")

iou_dict = {}

def labelStudioTokenize(text):
    return re.findall(r'\w+|[^\w\s]', text)

def pick_assignee(ann1,ann2):
    min_value = float('inf')
    index=0
    for i in range(4):
        if assignment_list[i] != ann1 and assignment_list[i] != ann2:
            if assignment_count[i] < min_value:
                min_value = assignment_count[i]
                index = i
    assignment_count[index] += 1
    return assignment_list[index]
            

def assign_annotation(obj_merged, index):
    file_handle = assignment_files[index]
    file_handle.write(f"{obj_merged}\n")
    return

def extract_batch_number(filename):
    """Extract the batch number from the filename."""
    return filename.split('|')[1].split('.')[1]

def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_kappa(matrix, batch_number):
    """Compute Cohen's Kappa."""
    total = np.sum(matrix)
    P_o = np.trace(matrix) / total  # Observed agreement
    P_e = np.sum(np.sum(matrix, axis=0) * np.sum(matrix, axis=1)) / (total ** 2)  # Expected agreement
    kappa = (P_o - P_e) / (1 - P_e) if (1 - P_e) != 0 else 0
    return kappa

def get_text_spans(text_length, annotations):
    """Convert annotations to a list of (start, end, label) tuples and add unannotated spans."""
    spans = []
    # Add annotated spans
    for ann in annotations:
        start = ann["value"]["start"]
        end = ann["value"]["end"]
        label = ann["value"]["labels"][0]
        text = ann["value"]["text"]
        spans.append((start, end, label, text))
    
    # Sort spans by start position
    spans.sort()
    
    # Add unannotated spans
    complete_spans = []
    current_pos = 0
    
    for start, end, label, text in spans:
        # If there's a gap before this annotation, add it as "Unannotated"
        if start > current_pos:
            complete_spans.append((current_pos, start - 1, "Unannotated", text))
        complete_spans.append((start, end, label, text))
        current_pos = end + 1
    
    # Add final unannotated span if necessary
    if current_pos < text_length:
        complete_spans.append((current_pos, text_length - 1, "Unannotated", ""))
    
    return complete_spans

def analyze_sequences(list1, list2, text_length):
    # Convert tuples to Range objects for easier handling
    ranges1 = [(start, end, label) for start, end, label, text in list1]
    ranges2 = [(start, end, label) for start, end, label, text in list2]
    list1 = [0 for _ in range(text_length+1)]
    list2 = [0 for _ in range(text_length+1)]
    for start, end, _ in ranges1:
        for i in range(start, end + 1):
            list1[i] = 1
    for start, end, _ in ranges2:
        for i in range(start, end + 1):
            list2[i] = 2
    overlap = 0
    union = 0
    for i in range(text_length):
        if list1[i] == 1 and list2[i] == 2:
            overlap += 1
        if list1[i] == 1 or list2[i] == 2:
            union += 1
    if union == 0:
        return 0
    return  round(overlap / union, 2)


def calculate_overlap(range1, range2):
    """Calculate the overlap between two ranges."""
    start = max(range1[0], range2[0])
    end = min(range1[1], range2[1])
    return max(0, end - start)

def find_best_match(source_tuple, target_list):
    """Find the tuple with maximum overlap from the target list."""
    max_overlap = 0
    best_match = None
    
    for target_tuple in target_list:
        overlap = calculate_overlap(
            (source_tuple[0], source_tuple[1]),
            (target_tuple[0], target_tuple[1])
        )
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = target_tuple
    
    return best_match

def analyze_match(source_tuple, matched_tuple):
    """Analyze the type of match between two tuples."""
    if not matched_tuple:
        return "no match"
    
    # Check if it's a full match
    is_full_match = (source_tuple[0] == matched_tuple[0] and 
                    source_tuple[1] == matched_tuple[1])
    
    # Check if labels match
    labels_match = source_tuple[2] == matched_tuple[2]
    
    if is_full_match:
        return "full match and label match" if labels_match else "full match and label mismatch"
    else:
        overlap = calculate_overlap(
            (source_tuple[0], source_tuple[1]),
            (matched_tuple[0], matched_tuple[1])
        )
        if overlap > 0:
            return "partial match and label match" if labels_match else "partial match and label mismatch"
        return "no match"

def get_final_result(results):
    """Determine the final result based on the new priority rules."""
    # Check if there are any partial matches or no matches
    has_partial = any(result.startswith("partial") for result in results)
    has_no_match = any(result == "no match" for result in results)
    has_full_match = any(result.startswith("full") for result in results)
    if not (has_partial or has_no_match):
        # All matches are full matches
        if any(result == "full match and label mismatch" for result in results):
            return "full match and label mismatch"
        return "full match and label match"
    
    # Check for partial matches
    if has_partial:
        if any(result == "partial match and label mismatch" for result in results):
            return "partial match and label mismatch"
        if any(result == "full match and label mismatch" for result in results):
            return "partial match and label mismatch"
        return "partial match and label match"
    if has_full_match:
        # All matches are full matches but some spans are unmatched
        if any(result == "full match and label mismatch" for result in results):
            return "partial match and label mismatch"
        return "partial match and label match"
    # If we get here, it must be no match
    return "no match"

def merge_mismatched_annotations(obj1, obj2, ann1, ann2):
    results = []
    prediction = {}
    # print(obj1)
    for i in obj1["annotations"][0]["result"]:
        result = {
            "value": i["value"],
            "id": uuid.uuid4().hex,
            "from_name": i["from_name"],
            "to_name": "text1",
            "type": i["type"],
        }
        results.append(result)
    for j in obj2["annotations"][0]["result"]:
        result = {
            "value": j["value"],
            "id": uuid.uuid4().hex,
            "from_name": j["from_name"],
            "to_name": "text2",
            "type": j["type"],
        }
        results.append(result)
    prediction["result"] = results
    prediction["score"] = 1
    prediction["model_version"] = "mismatch"
    predictions = [prediction]
    data = {
        "text1": obj1["data"]["text"],
        "text2": obj2["data"]["text"],
        "id": obj1["data"]["id"],
        "target": obj1["data"]["target"],
    }
    return {"data": data, "predictions": predictions}

def analyze_tuple_lists(list1, list2):
    """Analyze two lists of tuples and return the final result."""
    results = []
    results_with_range = []
    # Check list1 against list2
    for tuple1 in list1:
        best_match = find_best_match(tuple1, list2)
        result = analyze_match(tuple1, best_match)
        if not result == "no match":
            #print(result, tuple1[0],tuple1[1], tuple1[2], best_match[0], best_match[1], best_match[2])
            start = max(tuple1[0],best_match[0])
            end = min(tuple1[1],best_match[1])
            results_with_range.append((result,start,end, tuple1[3], best_match[3]))
            #print((result,start,end))
        results.append(result)
        
    # Check list2 against list1
    for tuple2 in list2:
        best_match = find_best_match(tuple2, list1)
        result = analyze_match(tuple2, best_match)
        if not result == "no match":
            #print(result, tuple2[0],tuple2[1], tuple2[2], best_match[0], best_match[1], best_match[2])
            start = max(tuple2[0],best_match[0])
            end = min(tuple2[1],best_match[1])
            results_with_range.append((result,start,end, tuple2[3], best_match[3]))
            #print((result,start,end))
        results.append(result)
    return get_final_result(results), results, results_with_range
no_label_match_span_mismatch_over_point_five = 0

def get_current_annotator(filename):
    """Extract annotator name from filename."""
    base = os.path.basename(filename)
    return base.split('|')[0]

def handle_annotation_mismatch(result, all_results, is_turkish, obj1, obj2, iou, file1, file2):
    global no_label_match_span_mismatch_over_point_five
    if result == "full match and label match":
        if is_turkish:
            full_match_file_tr.write(f"{obj1}\n")
        else:
            full_match_file_ar.write(f"{obj1}\n")
    elif result == "partial match and label match" and iou > 0.5:
        # {'id': 5775, 'annotations': [{'id': 818, 'completed_by': 3, 'result': [{'value': {'start': 12, 'end': 43, 'text': 'sözde Ermeni soykırımı yalanını', 'labels': ['Abartma, Genelleme, Yükleme, Çarpıtma']}, 'id': 'd303b11a-bba4-431e-a33d-09454a9ba1b3', 'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'prediction'}], 'was_cancelled': False, 'ground_truth': False, 'created_at': '2024-11-05T07:13:07.563391Z', 'updated_at': '2024-11-05T07:13:07.563422Z', 'draft_created_at': None, 'lead_time': 5.196, 'prediction': {'id': 41481, 'result': [{'id': 'd303b11a-bba4-431e-a33d-09454a9ba1b3', 'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'value': {'start': 12, 'end': 43, 'score': 1, 'text': 'sözde Ermeni soykırımı yalanını', 'labels': ['Abartma, Genelleme, Yükleme, Çarpıtma']}, 'error': None}], 'model_version': 'v1', 'created_ago': '1\xa0week, 1\xa0day', 'score': 1.0, 'cluster': None, 'neighbors': None, 'mislabeling': 0.0, 'created_at': '2024-10-27T20:42:26.545206Z', 'updated_at': '2024-10-27T20:42:26.545214Z', 'model': None, 'model_run': None, 'task': 5775, 'project': 810}, 'result_count': 0, 'unique_id': 'a955ec59-cc00-4e0c-a377-cdb29d4eedae', 'import_id': None, 'last_action': None, 'task': 5775, 'project': 810, 'updated_by': 3, 'parent_prediction': 41481, 'parent_annotation': None, 'last_created_by': None}], 'drafts': [], 'predictions': [41481], 'data': {'text': 'Öğrenciler, sözde Ermeni soykırımı yalanını belgelerle anlattı https://t.co/0yCkjdWOwI https://t.co/q6YT4YsUBT', 'id': '1658490375970131968', 'target': 'Hedef: Ermeni Karşıtı'}, 'meta': {}, 'created_at': '2024-10-27T20:42:26.537634Z', 'updated_at': '2024-11-05T07:13:07.624426Z', 'inner_id': 5, 'total_annotations': 1, 'cancelled_annotations': 0, 'total_predictions': 1, 'comment_count': 0, 'unresolved_comment_count': 0, 'last_comment_updated_at': None, 'project': 810, 'updated_by': 3, 'comment_authors': []}
        new_results = []
        # get overlapping parts of obj1["annotations"][0]["result"] and obj2["annotations"][0]["result"] which have öatching labels and intervals determined by the start and end values has overlap add intersection to new_results
        for i in obj1["annotations"][0]["result"]:
            for j in obj2["annotations"][0]["result"]:
                if i["value"]["labels"] == j["value"]["labels"]:
                    if i["value"]["start"] <= j["value"]["end"] and i["value"]["end"] >= j["value"]["start"]:
                        start = max(i["value"]["start"],j["value"]["start"])
                        end = min(i["value"]["end"],j["value"]["end"])
                        new_result = {
                            "value": {
                                "start": start,
                                "end": end,
                                "text": obj1["data"]["text"][start:end+1],
                                "labels": i["value"]["labels"]
                            },
                            "id": uuid.uuid4().hex,
                            "from_name": i["from_name"],
                            "to_name": "text",
                            "type": i["type"]
                        }
                        new_results.append(new_result)
        new_obj = obj1
        new_obj["annotations"][0]["result"] = new_results
        if is_turkish:
            pass
            full_match_file_tr.write(f"{new_obj}\n")
        else:
            pass
            full_match_file_ar.write(f"{new_obj}\n")
        no_label_match_span_mismatch_over_point_five += 1
    else: 
        annotator1 = get_current_annotator(file1)
        annotator2 = get_current_annotator(file2) 
        merged = merge_mismatched_annotations(obj1, obj2, annotator1, annotator2)
        if is_turkish:
            assignee_index = pick_assignee(annotator1,annotator2)
            assign_annotation(merged, assignment_list.index(assignee_index))
        else: 
            assign_annotation(merged, 4)
def compare_annotations(is_turkish, file1_data, file2_data, categories, batch_number,precision_recall_matrix,file1,file2):
    """Compare annotations between two JSON files and compute kappa."""
    matches = []
    # Add "Unannotated" to categories
    all_categories = categories + ["Unannotated"]
    category_to_index = {category: idx for idx, category in enumerate(all_categories)}
    matrix = np.zeros((len(all_categories), len(all_categories)))
    relaxed_kappa_matrix = np.zeros((len(all_categories), len(all_categories)))
    for obj1 in file1_data:
        for obj2 in file2_data:
            if obj1["data"]["id"] == obj2["data"]["id"]:
                text_length = len(obj1["data"]["text"])
                match_info = {
                    "id": obj1["id"], 
                    "text": obj1["data"]["text"],
                    "match_category": None,
                    "all_results": [],
                    "iou": 0,
                    "precision": [],
                    "recall": []
                }
                filename = "../annotation_stats/turkish_annotation_examples.txt" if is_turkish else "../annotation_stats/arabic_annotation_examples.txt"

                with open(filename, 'a', encoding='utf-8') as file:
                    file.write("-" * 100 + "\n")
                    file.write(f"Tweet Text: {obj1['data']['text']}\n")
                    file.write("*" * 10 + "\n")
                    file.write(f"Annotator 1 Spans\n")
                    for i in obj1["annotations"][0]["result"]:
                        file.write(f"{i['value']['text']}\n")
                        file.write(f"{i['value']['labels'][0]}\n")
                        file.write("." * 10 + "\n")
                    file.write("*" * 10 + "\n")
                    file.write(f"Annotator 2 Spans\n")
                    for j in obj2["annotations"][0]["result"]:
                        file.write(f"{j['value']['text']}\n")
                        file.write(f"{j['value']['labels'][0]}\n")
                        file.write("." * 10 + "\n")
                    file.write("*" * 10 + "\n")
                    file.write(f"Overlapping Annotations\n")
                    for i in obj1["annotations"][0]["result"]:
                        for j in obj2["annotations"][0]["result"]:
                            if i["value"]["labels"] == j["value"]["labels"]:
                                if i["value"]["start"] <= j["value"]["end"] or i["value"]["end"] >= j["value"]["start"]:
                                    start = max(i["value"]["start"], j["value"]["start"])
                                    end = min(i["value"]["end"], j["value"]["end"])
                                    overlap_text = obj1['data']['text'][start:end+1].strip()
                                    if overlap_text:  # Skip overlaps with only whitespaces
                                        file.write(f"{overlap_text}\n")
                                        file.write(f"{i['value']['labels'][0]}\n")
                                        file.write("." * 10 + "\n")
                    file.write("*" * 10 + "\n")
                    file.write(f"Overlapping Annotations with Category Mismatch\n")
                    for i in obj1["annotations"][0]["result"]:
                        for j in obj2["annotations"][0]["result"]:
                            if i["value"]["labels"] != j["value"]["labels"]:
                                if i["value"]["start"] <= j["value"]["end"] or i["value"]["end"] >= j["value"]["start"]:
                                    start = max(i["value"]["start"], j["value"]["start"])
                                    end = min(i["value"]["end"], j["value"]["end"])
                                    overlap_text = obj1['data']['text'][start:end+1].strip()
                                    if overlap_text:  # Skip overlaps with only whitespaces
                                        file.write(f"{overlap_text}\n")
                                        file.write(f"Annotator 1 Category: {i['value']['labels'][0]}\n")
                                        file.write(f"Annotator 2 Category: {j['value']['labels'][0]}\n")
                                        file.write("." * 10 + "\n")
                # Get all spans including unannotated for kappa calculation
                full_spans1 = get_text_spans(text_length, obj1["annotations"][0]["result"])
                full_spans2 = get_text_spans(text_length, obj2["annotations"][0]["result"])
                # Get only annotated spans for categorization
                spans1 = [(start, end, label, text) for start, end, label, text in full_spans1 if label != "Unannotated"]
                spans2 = [(start, end, label, text) for start, end, label, text in full_spans2 if label != "Unannotated"]
                iou = analyze_sequences(spans1, spans2, text_length)
                # Update confusion matrix using all spans (including unannotated)
                for start1, end1, label1, text in full_spans1:
                    for start2, end2, label2, text in full_spans2:
                        overlap_start = max(start1, start2)
                        overlap_end = min(end1, end2)
                        if overlap_start <= overlap_end:
                            overlap_length = len(labelStudioTokenize(obj1["data"]["text"][overlap_start:overlap_end+1]))
                            #overlap_length = overlap_end - overlap_start + 1
                            i = category_to_index[label1]
                            j = category_to_index[label2]
                            ## Relaxed Kappa
                            if iou > iou_threshold_for_full_match:
                                relaxed_kappa_matrix[i, i] += overlap_length
                            else:
                                relaxed_kappa_matrix[i, j] += overlap_length
                            matrix[i, j] += overlap_length
                            precision_recall_matrix[i, j] += overlap_length
                
                result, all_results, all_results_with_range = analyze_tuple_lists(spans1, spans2)
                # if result == "no match":
                #     if len(spans1) != 0 and len(spans2) != 0:
                #         print(obj1["data"]["id"])

                handle_annotation_mismatch(result, all_results, is_turkish, obj1, obj2, iou, file1, file2)
                match_info["match_category"] = result
                match_info["all_results"] = all_results
                match_info["all_results_with_range"] = all_results_with_range
                match_info["iou"] = iou
                iou_dict[iou] = iou_dict.get(iou, 0) + 1
                # if(obj1["data"]["id"] == "1657986183559802880"):
                #     print("obj1",obj1)
                #     print("obj2",obj2)
                #     print("match_info",match_info)
                matches.append(match_info)
    
    # Compute Kappa
    relaxed_kappa = compute_kappa(relaxed_kappa_matrix, batch_number)
    kappa = compute_kappa(matrix, batch_number)
    return matches, kappa, relaxed_kappa, matrix, relaxed_kappa_matrix

def calculate_micro_macro_metrics(matrix, exclude_last=True):
    """
    Calculate single precision and recall values for all categories,
    with an option to exclude the last category (micro- and macro-averaging).

    Parameters:
        matrix (np.ndarray): Confusion matrix (n_classes x n_classes).
        exclude_last (bool): If True, excludes the last category (row and column) from calculations.

    Returns:
        dict: A dictionary containing 'micro_precision', 'micro_recall', 
              'macro_precision', and 'macro_recall'.
    """
    # print(matrix)
    a = sum([sum(i) for i in matrix[:-1,:-1]])
    b = sum(matrix[:-1,-1])
    c = sum(matrix[-1,:-1])
    d = matrix[-1,-1]
    # print(a/(a+b))
    # print(a/(a+c))
    if exclude_last:
        true_positives = np.diag(matrix)[:-1]
        sum_cols = np.sum(matrix, axis=0)[:-1]
        sum_rows = np.sum(matrix, axis=1)[:-1]
    else:
        # True positives (diagonal elements)
        true_positives = np.diag(matrix)
        # Sum over columns (predicted positives) and rows (actual positives)
        sum_cols = np.sum(matrix, axis=0)
        sum_rows = np.sum(matrix, axis=1)

    # Micro-Averaging
    total_true_positives = np.sum(true_positives)
    total_predicted_positives = np.sum(sum_cols)
    total_actual_positives = np.sum(sum_rows)

    micro_precision = total_true_positives / total_predicted_positives if total_predicted_positives != 0 else 0
    micro_recall = total_true_positives / total_actual_positives if total_actual_positives != 0 else 0

    # Macro-Averaging
    # Individual class precision and recall (handling division by zero)
    precision_per_class = np.divide(
        true_positives, sum_cols, out=np.zeros_like(sum_cols, dtype=float), where=sum_cols != 0
    )
    recall_per_class = np.divide(
        true_positives, sum_rows, out=np.zeros_like(sum_rows, dtype=float), where=sum_rows != 0
    )

    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_detection_precision": a / (a + c) if a + c != 0 else 0,
        "micro_detection_recall": a / (a + b) if a + b != 0 else 0,
    }


def calculate_precision_recall(matrix):
    # Sum over columns and rows
    sum_cols = np.sum(matrix, axis=0)
    sum_rows = np.sum(matrix, axis=1)
    
    # Precision: Handle division by zero
    precision = np.divide(
        np.diag(matrix), sum_cols, out=np.zeros_like(sum_cols, dtype=float), where=sum_cols != 0
    )
    
    # Recall: Handle division by zero
    recall = np.divide(
        np.diag(matrix), sum_rows, out=np.zeros_like(sum_rows, dtype=float), where=sum_rows != 0
    )
    
    return precision, recall


def process_files(file_dir, categories, precision_recall_matrix, is_turkish):
    """Process files in a directory to find matches and compute kappa."""
    # Group files by batch number
    files_by_batch = defaultdict(list)
    for file_name in os.listdir(file_dir):
        if file_name.endswith('.json'):
            batch_number = extract_batch_number(file_name)
            files_by_batch[batch_number].append(os.path.join(file_dir, file_name))
    all_categories = categories + ["Unannotated"]
    overall_matrix = np.zeros((len(all_categories), len(all_categories)))
    overall_relaxed_kappa_matrix = np.zeros((len(all_categories), len(all_categories)))
    # Compare files within each batch
    results = {}
    for batch_number, files in files_by_batch.items():
        if len(files) == 2:  # Only process batches with exactly two files
            file1_data = load_json(files[0])
            file2_data = load_json(files[1])
            matches, kappa, relaxed_kappa, matrix, relaxed_matrix = compare_annotations(is_turkish,file1_data, file2_data, categories, batch_number,precision_recall_matrix,files[0], files[1])
            batch_precision, batch_recall = calculate_precision_recall(matrix)
            overall_matrix += matrix
            overall_relaxed_kappa_matrix += relaxed_matrix
            results[batch_number] = {
                "matches": matches, 
                "kappa": kappa,
                "relaxed_kappa": relaxed_kappa,
                "confusion_matrix": matrix.tolist(),  # Convert to list for JSON serialization
                "precision": batch_precision,
                "recall": batch_recall
            }
    overall_relaxed_kappa = compute_kappa(overall_relaxed_kappa_matrix, "all")
    overall_kappa = compute_kappa(overall_matrix, "all")
    for batch in results:
        results[batch]["overall_kappa"] = overall_kappa
        results[batch]["overall_relaxed_kappa"] = overall_relaxed_kappa
    micro_macro_metrics_exlcude_unannotated = calculate_micro_macro_metrics(precision_recall_matrix, exclude_last=True)
    micro_macro_metrics = calculate_micro_macro_metrics(precision_recall_matrix, exclude_last=False)
    
    precision_recall_results = calculate_precision_recall(precision_recall_matrix)

    precision_recall_results_micro_excluded = micro_macro_metrics_exlcude_unannotated["micro_precision"], micro_macro_metrics_exlcude_unannotated["micro_recall"]
    precision_recall_results_macro_excluded = micro_macro_metrics_exlcude_unannotated["macro_precision"], micro_macro_metrics_exlcude_unannotated["macro_recall"]
    precision_recall_results_micro = micro_macro_metrics["micro_precision"], micro_macro_metrics["micro_recall"], micro_macro_metrics["micro_detection_precision"], micro_macro_metrics["micro_detection_recall"]
    precision_recall_results_macro = micro_macro_metrics["macro_precision"], micro_macro_metrics["macro_recall"]
    return results, precision_recall_results, precision_recall_results_micro, precision_recall_results_macro, precision_recall_results_micro_excluded, precision_recall_results_macro_excluded

file_directory1 = "../annotations/turkish"
file_directory2 = "../annotations/arabic"
categories1 = ["Dışlama; Ayrımcı Söylem","Düşmanlık, Savaş, Saldırı, Öldürme, Yaralama Tehditi","Simgeleştirme","Abartma, Genelleme, Yükleme, Çarpıtma","Küfür, Hakaret, Aşağılama, İnsandışılaştırma"]  
categories2 = ["Exclusive/Discriminatory Discourse","Exaggeration; Generalization; Attribution; Distortion","Threat of Enmity; War; Attack; Murder; or Harm","Symbolization","Swearing; Insult; Defamation; Dehumanization"]

all_categories1 = categories1 + ["Unannotated"]
precision_recall_matrix_1 = np.zeros((len(all_categories1), len(all_categories1)))
results1,precision_recall_results1, micro_precision_recall_results1, macro_precision_recall_results1, micro_precision_recall_excluded_results1, macro_precision_recall_excluded_results1 = process_files(file_directory1, categories1, precision_recall_matrix_1, True)

all_categories2 = categories2 + ["Unannotated"]
precision_recall_matrix_2 = np.zeros((len(all_categories2), len(all_categories2)))
results2,precision_recall_results2, micro_precision_recall_results2, macro_precision_recall_results2, micro_precision_recall_excluded_results2, macro_precision_recall_excluded_results2  = process_files(file_directory2, categories2, precision_recall_matrix_2, False)

def write_analysis_results(f, language, categories, results, precision_recall_results, micro_precision_recall_results, macro_precision_recall_results, precision_recall_results_micro_excluded, precision_recall_results_macro_excluded):
    all_categories = categories + ["Unannotated"]
    f.write(f"Kappa for {language}: {results.get('2')['overall_kappa']:.2f}\n") 
    f.write(f"Relaxed Kappa for {language}: {results.get('2')['overall_relaxed_kappa']:.2f}\n")
    # Write precision and recall for each category
    f.write(f"Precision and Recall for {language} annotations:\n")
    f.write("-" * 50 + "\n\n")

    f.write(f"Micro-Averaged Precision: {micro_precision_recall_results[0]:.2f}\n\n")
    f.write(f"Micro-Averaged Recall: {micro_precision_recall_results[1]:.2f}\n\n")
    f.write(f"Macro-Averaged Precision: {macro_precision_recall_results[0]:.2f}\n\n")
    f.write(f"Macro-Averaged Recall: {macro_precision_recall_results[1]:.2f}\n\n")

    f.write(f"Micro-Averaged Precision (Excluding Unannotated): {precision_recall_results_micro_excluded[0]:.2f}\n\n")
    f.write(f"Micro-Averaged Recall (Excluding Unannotated): {precision_recall_results_micro_excluded[1]:.2f}\n\n")
    f.write(f"Macro-Averaged Precision (Excluding Unannotated): {precision_recall_results_macro_excluded[0]:.2f}\n\n")
    f.write(f"Macro-Averaged Recall (Excluding Unannotated): {precision_recall_results_macro_excluded[1]:.2f}\n\n")

    f.write(f"Micro-Averaged Precision for Detection:{micro_precision_recall_results[2]:.2f}\n\n")
    f.write(f"Micro-Averaged Recall for Detection:{micro_precision_recall_results[3]:.2f}\n\n")

    for i in range(len(precision_recall_results[0])):
        f.write(f"Precision for {all_categories[i]}: {precision_recall_results[0][i]}\n\n")
    for i in range(len(precision_recall_results[1])):
        f.write(f"Recall for {all_categories[i]}: {precision_recall_results[1][i]}\n\n")

    # Write Kappa scores for all batches
    f.write(f"Cohen's Kappa Scores for {language}:\n")
    f.write("-" * 50 + "\n")
    for batch in sorted([int(x) for x in results.keys()]):
        f.write(f"Batch {batch}: {results.get(str(batch))['kappa']:.2f}\n")
        f.write(f"Relaxed Kappa for Batch {batch}: {results.get(str(batch))['relaxed_kappa']:.2f}\n")
    f.write("\n")

def write_detailed_batch_info(f, language, categories, results):
    all_categories = categories + ["Unannotated"]
    
    f.write(f"Detailed Batch Information for {language}:\n")
    f.write("-" * 50 + "\n")
    
    for batch in sorted([int(x) for x in results.keys()]):
        batch_str = str(batch)
        f.write(f"\nBatch {batch}:\n")
        if language == "Turkish":  # Add extra formatting for Turkish
            f.write("*" * 50 + "\n")
            
        for i in range(len(results[batch_str]['precision'])):
            f.write(f"Precision for {all_categories[i]}: {results[batch_str]['precision'][i]}\n")
            if language == "Turkish":  # Add extra newline for Turkish
                f.write("\n")
                
        for i in range(len(results[batch_str]['recall'])):
            f.write(f"Recall for {all_categories[i]}: {results[batch_str]['recall'][i]}\n")
            if language == "Turkish":  # Add extra newline for Turkish
                f.write("\n")
                
        if language == "Turkish":  # Add extra formatting for Turkish
            f.write("*" * 50 + "\n")

        # Count categories in this batch
        batch_categories = {
            "full match and label match": 0,
            "full match and label mismatch": 0,
            "partial match and label match": 0,
            "partial match and label mismatch": 0,
            "no match": 0
        }
        
        for match in results[batch_str]["matches"]:
            batch_categories[match["match_category"]] += 1
        
        # Write batch category statistics
        batch_total = sum(batch_categories.values())
        f.write(f"Category Statistics:\n")
        for category, count in batch_categories.items():
            percentage = (count / batch_total * 100) if batch_total > 0 else 0
            f.write(f"  {category}: {count} ({percentage:.2f}%)\n")
        
        # Write detailed match information
        f.write("\nDetailed Matches:\n")
        for match in results[batch_str]["matches"]:
            f.write(f"\n  Text ID: {match['id']}\n")
            f.write(f"  Overall Category: {match['match_category']}\n")
            f.write(f"  Text: {match['text']}\n")
            f.write(f"  IOU: {match['iou']:.2f}\n")
            for i in range(len(match['precision'])):
                f.write(f"Precision for {all_categories[i]}: {match['precision'][i]}\n")
            for i in range(len(match['recall'])):
                f.write(f"Recall for {all_categories[i]}: {match['recall'][i]}\n")
            f.write("\n")
            # f.write(f"  Match Results:\n")
            # for result in match["all_results"]:
            #     f.write(f"    {result}\n")
            # f.write("\n")
            # for result in match["all_results_with_range"]:
            #     f.write(f"    {result}\n")
            # f.write("\n")

def write_overall_statistics(f, results1, results2):
    f.write("Overall Category Statistics:\n")
    f.write("-" * 50 + "\n")
    category_counts = {
        "full match and label match": 0,
        "full match and label mismatch": 0,
        "partial match and label match": 0,
        "partial match and label mismatch": 0,
        "no match": 0
    }

    # Count occurrences for both languages
    for results in [results1, results2]:
        for batch in results.values():
            for match in batch["matches"]:
                category_counts[match["match_category"]] += 1

    total_texts = sum(category_counts.values())
    for category, count in category_counts.items():
        percentage = (count / total_texts * 100) if total_texts > 0 else 0
        f.write(f"{category}: {count} ({percentage:.2f}%)\n")
    f.write(f"Total Count: {total_texts}\n")
    f.write("\n")

def write_language_results(filename, language, categories, results, precision_recall_results, micro_precision_recall_results, macro_precision_recall_results, precision_recall_results_micro_excluded, precision_recall_results_macro_excluded):
    with open(filename, "w", encoding='utf-8') as f:
        write_analysis_results(f, language, categories, results, precision_recall_results, micro_precision_recall_results, macro_precision_recall_results, precision_recall_results_micro_excluded, precision_recall_results_macro_excluded)
        write_detailed_batch_info(f, language, categories, results)

# Main execution
# Write separate files for Turkish and Arabic results
write_language_results("../annotation_stats/turkish_output.txt", "Turkish", categories1, results1, precision_recall_results1, micro_precision_recall_results1, macro_precision_recall_results1, micro_precision_recall_excluded_results1, macro_precision_recall_excluded_results1)
write_language_results("../annotation_stats/arabic_output.txt", "Arabic", categories2, results2, precision_recall_results2, micro_precision_recall_results2, macro_precision_recall_results2, micro_precision_recall_excluded_results2, macro_precision_recall_excluded_results2)  

# Write combined statistics to a separate file
with open("../annotation_stats/combined_statistics.txt", "w", encoding='utf-8') as f:
    write_overall_statistics(f, results1, results2)

# Write histogram data
with open("../annotation_stats/histogram.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Label', 'IOU'])
    for label in sorted(iou_dict.keys()):
        writer.writerow([label, iou_dict[label]])

for i in assignment_files:
    i.close()

for j in range(4):
    # print(f"number of assignments for {assignment_list[j]}:"+str(assignment_count[j]))
    pass

full_match_file_ar.close()
full_match_file_tr.close()
# print("patial matches with iou>0.5:"+str(no_label_match_span_mismatch_over_point_five))