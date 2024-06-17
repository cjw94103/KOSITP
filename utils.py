import json

def save_json(filename, data):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def load_json_file(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)
    return data

def load_jsonl_file(file_path, encoding='utf-8'):
    data = []
    with open(file_path, 'r', encoding=encoding) as file:
        for line in file:
            data.append(json.loads(line))
    return data

def save_jsonl(filename, data_list):
    with open(filename, 'w') as f:
        for data in data_list:
            json.dump(data, f)
            f.write('\n')