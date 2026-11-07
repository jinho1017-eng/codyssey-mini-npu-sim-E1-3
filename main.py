import json
from npu import compute_mac_score, judge_scores, measure_mac_time, normalize_label, make_cross, make_x


def read_grid(label, n):
    """n줄을 입력받아 n x n 2차원 리스트로 반환한다. 형식이 틀리면 재입력을 받는다."""
    while True:
        print(f"{label} ({n}줄 입력, 공백 구분)")
        rows = []
        error = False
        for _ in range(n):
            line = input().split()
            if len(line) != n:
                error = True
                break
            try:
                row = [float(value) for value in line]
            except ValueError:
                error = True
                break
            rows.append(row)
        if error:
            print(f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요.")
            continue
        return rows


def run_mode1():
    """사용자 입력(3x3) 모드: 필터 A, B, 패턴을 입력받아 MAC 점수를 계산한다."""
    filter_a = read_grid("필터 A", 3)
    filter_b = read_grid("필터 B", 3)
    pattern = read_grid("패턴", 3)

    score_a = compute_mac_score(pattern, filter_a)
    score_b = compute_mac_score(pattern, filter_b)

    print(f"A 점수: {score_a}")
    print(f"B 점수: {score_b}")

    result = judge_scores(score_a, score_b)
    if result == "TIE":
        print("판정: 판정 불가 (|A-B| < 1e-9)")
    else:
        print(f"판정: {result}")

    avg_time = measure_mac_time(pattern, filter_a)
    print(f"연산 시간(평균/10회): {avg_time:.3f} ms")


def load_data(path="data.json"):
    """data.json 파일을 읽어서 딕셔너리로 반환한다."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_filters(raw_filters):
    """{"size_5": {"cross": [...], "x": [...]}} 형태를
    {5: {"Cross": [...], "X": [...]}} 형태로 변환한다."""
    filters = {}
    for size_key, label_map in raw_filters.items():
        n = int(size_key.split("_")[1])
        filters[n] = {}
        for label_key, grid in label_map.items():
            standard_label = normalize_label(label_key)
            filters[n][standard_label] = grid
    return filters


def run_mode2(path="data.json"):
    """data.json 분석 모드: 패턴마다 MAC 판정을 수행하고 PASS/FAIL을 출력한다."""
    raw = load_data(path)
    filters = build_filters(raw["filters"])

    results = []
    for pattern_key, pattern_info in raw["patterns"].items():
        pattern_input = pattern_info["input"]
        size = int(pattern_key.split("_")[1])

        print(f"--- {pattern_key} ---")

        if size not in filters:
            print(f"FAIL: size_{size} 필터가 없습니다.")
            results.append((pattern_key, False, f"size_{size} 필터 없음"))
            continue

        if len(pattern_input) != size or any(len(row) != size for row in pattern_input):
            print("FAIL: 패턴 크기가 필터 크기와 일치하지 않습니다.")
            results.append((pattern_key, False, "패턴 크기가 필터 크기와 일치하지 않음"))
            continue

        cross_score = compute_mac_score(pattern_input, filters[size]["Cross"])
        x_score = compute_mac_score(pattern_input, filters[size]["X"])

        result = judge_scores(cross_score, x_score)
        if result == "A":
            verdict = "Cross"
        elif result == "B":
            verdict = "X"
        else:
            verdict = "UNDECIDED"

        expected = normalize_label(pattern_info["expected"])

        print(f"Cross 점수: {cross_score}")
        print(f"X 점수: {x_score}")

        passed = (verdict == expected)
        print(f"판정: {verdict} | expected: {expected} | {'PASS' if passed else 'FAIL'}")

        reason = "" if passed else f"판정 불일치 (판정={verdict}, expected={expected})"
        results.append((pattern_key, passed, reason))

    return results


def run_performance_table(sizes=(3, 5, 13, 25)):
    """크기별 MAC 연산 평균 시간(ms)과 연산 횟수(N²)를 표로 출력한다."""
    print(f"{'크기':<10}{'평균 시간(ms)':<15}{'연산 횟수':<10}")
    print("-" * 35)
    for n in sizes:
        cross = make_cross(n)
        x = make_x(n)
        avg_time = measure_mac_time(cross, x)
        print(f"{n}x{n:<8}{avg_time:<15.3f}{n * n:<10}")


def print_summary(results):
    """results: [(pattern_key, passed, reason), ...] 리스트로 총/통과/실패 요약을 출력한다."""
    total = len(results)
    passed_count = sum(1 for _, passed, _ in results if passed)
    failed_count = total - passed_count

    print(f"총 테스트: {total}개")
    print(f"통과: {passed_count}개")
    print(f"실패: {failed_count}개")

    if failed_count > 0:
        print("실패 케이스:")
        for pattern_key, passed, reason in results:
            if not passed:
                print(f"- {pattern_key}: {reason}")

def main():
    print("=== Mini NPU Simulator ===")
    print()
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")

    while True:
        choice = input("선택: ").strip()
        if choice in ("1", "2"):
            break
        print("1 또는 2를 입력하세요.")

    if choice == "1":
        run_mode1()
    else:
        results = run_mode2()
        run_performance_table()
        print_summary(results)


if __name__ == "__main__":
    main()