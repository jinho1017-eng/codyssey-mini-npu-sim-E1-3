"""Mini NPU Simulator - MAC 연산 핵심 로직."""
import time


def compute_mac_score(pattern, filter_):
    """패턴과 필터를 위치별로 곱한 뒤 모두 더한 MAC 점수를 반환한다."""
    n = len(pattern)
    score = 0
    for i in range(n):
        for j in range(n):
            score += pattern[i][j] * filter_[i][j]
    return score


def normalize_label(label):
    """다양한 라벨 표기를 표준 라벨(Cross/X)로 정규화한다."""
    key = str(label).strip().lower()
    if key in ("+", "cross"):
        return "Cross"
    if key == "x":
        return "X"
    raise ValueError(f"알 수 없는 라벨입니다: {label}")


def judge_scores(score_a, score_b, epsilon=1e-9):
    """두 점수를 epsilon 기준으로 비교해 'A', 'B', 'TIE' 중 하나를 반환한다."""
    if abs(score_a - score_b) < epsilon:
        return "TIE"
    if score_a > score_b:
        return "A"
    return "B"


def measure_mac_time(pattern, filter_, repeats=10):
    """MAC 연산을 repeats번 반복 측정하여 평균 시간(ms)을 반환한다."""
    start = time.perf_counter()
    for _ in range(repeats):
        compute_mac_score(pattern, filter_)
    elapsed = time.perf_counter() - start
    return (elapsed / repeats) * 1000

def make_cross(n):
    """n x n 십자가(Cross) 패턴을 생성한다."""
    center = n // 2
    return [[1 if i == center or j == center else 0 for j in range(n)] for i in range(n)]


def make_x(n):
    """n x n X 패턴을 생성한다."""
    return [[1 if i == j or i + j == n - 1 else 0 for j in range(n)] for i in range(n)]