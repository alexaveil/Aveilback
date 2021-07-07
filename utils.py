import json

def save_json(data, path):
    with open(path, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data
