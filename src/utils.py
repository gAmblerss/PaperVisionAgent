import json
import base64

def load_json(json_path):
    with open(json_path,"r",encoding="utf-8") as f:
        return json.load(f)

def load_text(json_path):
    with open(json_path,"r",encoding="utf-8") as f:
        return f.read()

def encode_image_to_base64(image_path:str)->str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8") #把二进制图片数据变成可放进文本请求体里的字符串

