import os
import json

def filter_annotations(folder_path, output_file_path):

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        none = 0
        one = 0
        two = 0
        neither = 0
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                for obj in data:
                    result = obj['annotations'][0]['result'] 
                    filtered_result = []
                    choice = None
                    for item in result:
                        if item['type'] == 'choices':
                            if 'Annotation 1' in item['value']['choices']:
                                choice = "text1"
                            elif 'Annotation 2' in item['value']['choices']:
                                choice = "text2"
                            elif 'Neither' in item['value']['choices']:
                                choice = "Neither"
                            else:
                                print("error")
                    if choice is None or choice == "Neither":
                        if choice is None:
                            none += 1
                        elif choice == "Neither":
                            neither += 1
                        continue
                    else:
                        if choice == "text1":
                            one += 1
                            filtered_result = [item for item in result if item["to_name"] == "text1" and item["value"].get("choices",-1) == -1]
                        elif choice == "text2":
                            two += 1
                            filtered_result = [item for item in result if item["to_name"] == "text2"]
                        if choice == "text1" or choice == "text2":
                            # Update the result in a safe manner

                            obj["annotations"][0]["result"] = filtered_result
                            obj["data"]["text"] = obj["data"]["text1"]
                            # print("*******************************")
                            # print(obj["annotations"][0]["result"])
                            # print("-----------------------------")
                            # print(filtered_result)
                            # print("*******************************")
                            # Write the updated object to the output file
                            json_line = json.dumps(obj, ensure_ascii=False)
                            output_file.write(json_line + '\n')
        # print(f"None: {none}")
        # print(f"One: {one}")
        # print(f"Two: {two}")
        # print(f"Neither: {neither}")
    # replace null with None and false with False
    with open(output_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            line = line.replace("null", "None")
            line = line.replace("false", "False")
            line = line.replace("true", "True")
            file.write(line)

# Define folder paths
turkish_folder = '../handled_mismatch/turkish/'
arabic_folder = '../handled_mismatch/arabic/'

# Filter annotations in both folders
filter_annotations(turkish_folder,"../handled_mismatch/merged_turkish/handled_mismatch_turkish.txt")
filter_annotations(arabic_folder,"../handled_mismatch/merged_arabic/handled_mismatch_arabic.txt")


