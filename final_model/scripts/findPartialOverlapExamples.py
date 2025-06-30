import json

def extract_spans(results, to_name_filter=None):
    """Extract spans as dictionaries with start, end, and text."""
    return sorted([
        {
            "start": r["value"]["start"],
            "end": r["value"]["end"],
            "text": r["value"]["text"]
        }
        for r in results
        if to_name_filter is None or r["to_name"] == to_name_filter
    ], key=lambda x: (x["start"], x["end"]))

def span_overlap(span1, span2):
    """Check if two spans overlap."""
    return not (span1["end"] <= span2["start"] or span2["end"] <= span1["start"])

def find_common_spans(spans1, spans2, full_text):
    common = []
    for s1 in spans1:
        for s2 in spans2:
            if span_overlap(s1, s2):
                start_overlap = max(s1["start"], s2["start"])
                end_overlap = min(s1["end"], s2["end"])
                overlap_text = full_text[start_overlap:end_overlap].strip()
                if overlap_text and overlap_text not in common:
                    common.append(overlap_text)
    return sorted(common)

def process_file(input_filename, output_filename):
    with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if not line.strip():
                continue
            line = line.replace("null", "None").replace("false", "False").replace("true", "True")
            try:
                data = eval(line.strip())

                text = data["data"]["text"]
                annotations = data["annotations"][0]

                annotator_results = annotations.get("result", [])
                prediction_results = annotations.get("prediction", {}).get("result", [])

                text1_spans = extract_spans(prediction_results, "text1")
                text2_spans = extract_spans(prediction_results, "text2")
                gt_spans = extract_spans(annotator_results)

                text1_annots = [s["text"] for s in text1_spans]
                text2_annots = [s["text"] for s in text2_spans]
                gt_annots = [s["text"] for s in gt_spans]

                common_annots = find_common_spans(text1_spans, text2_spans, text)

                # Skip if total match or no overlap
                if set(text1_annots) == set(text2_annots) or not common_annots:
                    continue

                # Write lines with explanations
                outfile.write(f'Tweet: {text}\n')
                outfile.write(f'Different spans are separated by " ### "\n')
                outfile.write(f'Annotations of 1. annotator: {" ### ".join(text1_annots)}\n')
                outfile.write(f'Annotations of 2. annotator: {" ### ".join(text2_annots)}\n')
                outfile.write(f'Ground truth annotations: {" ### ".join(gt_annots)}\n')
                outfile.write(f'Common annotations: {" ### ".join(common_annots)}\n')
                outfile.write('----------\n')

            except Exception as e:
                print(f"Error processing line: {e}")
                continue

# Run the function
input_file = '../handled_mismatch/merged_turkish/handled_mismatch_turkish.txt'
output_file = '../partial_overlap_examples/partial_overlap_examples.txt'
process_file(input_file, output_file)
