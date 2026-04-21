import pandas as pd
import ast
import pprint
import sys
from pathlib import Path

def build_semantic_dicts(xlsx_path: str):
    path = Path(xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案: {xlsx_path}")

    df = pd.read_excel(path)

    semantic_dicts = {
        "colors": {},
        "indices": {},
    }

    for _, row in df.iterrows():
        name = str(row["Name"]).strip()
        index = int(row["Unnamed: 0"])

        # 把 "(120, 120, 120)" 轉成 [120, 120, 120]
        color_tuple = ast.literal_eval(row["Color_Code (R,G,B)"])
        color_list = list(color_tuple)

        semantic_dicts["colors"][name] = [color_list]
        semantic_dicts["indices"][name] = index

    return semantic_dicts


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python extract_xlsx.py <xlsx檔案路徑>")
        sys.exit(1)

    xlsx_path = sys.argv[1]
    result = build_semantic_dicts(xlsx_path)

    print("SEMANTIC_DICTS = ", end="")
    pprint.pprint(result, sort_dicts=False)