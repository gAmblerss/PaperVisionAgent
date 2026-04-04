import json

def load_json(json_path):
    with open(json_path,"r",encoding="utf-8") as f:
        return json.load(f)
def load_text(json_path):
    with open(json_path,"r",encoding="utf-8") as f:
        return f.read()