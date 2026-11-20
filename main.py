import json
from pathlib import Path
from npu import compute_mac_score, judge_scores, measure_mac_time, normalize_label, make_cross, make_x, compare_optimization

DATA_PATH = Path(__file__).with_name("data.json")


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


def print_grid(title, grid):
    """저장된 격자를 다시 출력해 입력값이 제대로 저장됐는지 확인시킨다."""
    print(f"{title} 저장 완료:")
    for row in grid:
        print(" ".join(f"{value:g}" for value in row))


def run_mode1():
    """사용자 입력(3x3) 모드: 필터 A, B, 패턴을 입력받아 MAC 점수를 계산한다."""
    use_generated = input("자동 생성된 패턴을 사용하시겠습니까? (y/n): ").strip().lower()

    if use_generated == "y":
        while True:
            size_text = input("패턴 크기 N 입력: ").strip()
            if size_text.isdigit() and int(size_text) >= 1:
                n = int(size_text)
                break
            print("1 이상의 정수를 입력하세요.")

        filter_a = make_cross(n)
        filter_b = make_x(n)
        print(f"필터 A(Cross, {n}x{n})와 필터 B(X, {n}x{n})가 자동 생성되었습니다.")

        while True:
            pattern_choice = input("패턴으로 사용할 모양 선택 (cross/x): ").strip().lower()
            if pattern_choice in ("cross", "x"):
                break
            print("cross 또는 x를 입력하세요.")
        pattern = make_cross(n) if pattern_choice == "cross" else make_x(n)
    else:
        filter_a = read_grid("필터 A", 3)
        filter_b = read_grid("필터 B", 3)
        print_grid("필터 A", filter_a)
        print_grid("필터 B", filter_b)
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
    print(f"연산 횟수(N²): {len(pattern) ** 2}")


def load_data(path=DATA_PATH):
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


def is_square_grid(grid, size):
    return isinstance(grid, list) and len(grid) == size and all(
        isinstance(row, list) and len(row) == size for row in grid
    )


def run_mode2(path=DATA_PATH):
    """data.json 분석 모드: 패턴마다 MAC 판정을 수행하고 PASS/FAIL을 출력한다."""
    raw = load_data(path)
    filters = build_filters(raw["filters"])

    results = []
    for pattern_key, pattern_info in raw["patterns"].items():
        print(f"--- {pattern_key} ---")

        try:
            size = int(pattern_key.split("_")[1])
        except (IndexError, ValueError):
            print("FAIL: 패턴 키 형식이 올바르지 않습니다. (예: size_5_1)")
            results.append((pattern_key, False, "패턴 키 형식 오류"))
            continue

        missing = [
            key for key in ("input", "expected")
            if not isinstance(pattern_info, dict) or key not in pattern_info
        ]
        if missing:
            names = ", ".join(missing)
            print(f"FAIL: 패턴에 필요한 항목이 없습니다 ({names}).")
            results.append((pattern_key, False, f"항목 누락: {names}"))
            continue

        pattern_input = pattern_info["input"]

        if size not in filters:
            print(f"FAIL: size_{size} 필터가 없습니다.")
            results.append((pattern_key, False, f"size_{size} 필터 없음"))
            continue

        if not is_square_grid(pattern_input, size):
            print("FAIL: 패턴 크기가 필터 크기와 일치하지 않습니다.")
            results.append((pattern_key, False, "패턴 크기가 필터 크기와 일치하지 않음"))
            continue

        cross_filter = filters[size].get("Cross")
        x_filter = filters[size].get("X")
        invalid_filters = [
            label
            for label, grid in (("Cross", cross_filter), ("X", x_filter))
            if not is_square_grid(grid, size)
        ]
        if invalid_filters:
            labels = ", ".join(invalid_filters)
            reason = f"필터 누락 또는 크기 불일치: {labels} (기대 크기: {size}×{size})"
            print(f"FAIL: {reason}")
            results.append((pattern_key, False, reason))
            continue

        cross_score = compute_mac_score(pattern_input, cross_filter)
        x_score = compute_mac_score(pattern_input, x_filter)

        result = judge_scores(cross_score, x_score)
        if result == "A":
            verdict = "Cross"
        elif result == "B":
            verdict = "X"
        else:
            verdict = "UNDECIDED"

        try:
            expected = normalize_label(pattern_info["expected"])
        except ValueError as error:
            print(f"FAIL: {error}")
            results.append((pattern_key, False, str(error)))
            continue

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
        size_label = f"{n}x{n}"
        print(f"{size_label:<10}{avg_time:<15.3f}{n * n:<10}")


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


def run_optimization_comparison(sizes=(3, 5, 13, 25)):
    """크기별로 2차원 방식과 1차원 방식의 평균 연산 시간을 표로 비교 출력한다."""
    print(f"{'크기':<10}{'2차원(ms)':<15}{'1차원(ms)':<15}")
    print("-" * 40)
    for n in sizes:
        time_2d, time_1d = compare_optimization(n)
        size_label = f"{n}x{n}"
        print(f"{size_label:<10}{time_2d:<15.3f}{time_1d:<15.3f}")

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
        try:
            results = run_mode2()
        except KeyError as error:
            print(f"data.json에 필요한 항목이 없습니다: {error}")
            print("data.json을 확인하거나, 프로젝트 폴더에서 python generate_data.py로 다시 만들어 주세요.")
            return
        except (OSError, ValueError) as error:
            print(f"data.json을 처리할 수 없습니다: {error}")
            print("data.json을 확인하거나, 프로젝트 폴더에서 python generate_data.py로 다시 만들어 주세요.")
            return
        run_performance_table()
        print_summary(results)


if __name__ == "__main__":
    main()