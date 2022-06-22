import os
import json


def get_dataset_path(key):
    return json.load(open(os.path.join(os.path.dirname(__file__), 'paths.json')))[key]
