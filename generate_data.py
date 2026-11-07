"""data.json 생성 스크립트 (1회 실행용)."""
import json
from npu import make_cross,make_x

def build_data():
    sizes = [5, 13, 25]
    data = {"filters": {}, "patterns": {}}

    for n in sizes:
        cross = make_cross(n)
        x = make_x(n)
        data["filters"][f"size_{n}"] = {"cross": cross, "x": x}

        data["patterns"][f"size_{n}_1"] = {"input": cross, "expected": "+"}
        data["patterns"][f"size_{n}_2"] = {"input": x, "expected": "x"}

    return data


if __name__ == "__main__":
    data = build_data()
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("data.json 생성 완료")