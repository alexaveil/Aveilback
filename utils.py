import json
import os
from datetime import datetime
import pickle

def save_json(data, path):
    with open(path, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def split_filename(path):
    filename = os.path.basename(path)
    dir = os.path.dirname(path)
    file, ext = os.path.splitext(filename)
    return dir, file, ext

def now2string(flatten=False):
    now = datetime.now()
    # dd/mm/YY H:M:S
    return now.strftime("%d/%m/%Y %H:%M:%S" if not flatten else "%d_%m_%Y__%H_%M_%S")
    
def save_pickle(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data
