from transformers import AutoTokenizer
import json
from datasets import Dataset

def load_data(path):
    """
    JSON ファイルを読み込み、datasets.Dataset オブジェクトとして返す。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # JSON ファイルをリスト形式で読み込む

    # リストを辞書形式に変換
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        dict_data = {key: [d[key] for d in data] for key in data[0].keys()}
        return Dataset.from_dict(dict_data)  # 辞書を datasets.Dataset に変換
    else:
        raise ValueError("Invalid data format in JSON file. Expected a list of dictionaries.")