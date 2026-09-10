from typing import List


def select_indices(n: int, max_items: int) -> List[int]:
    """Evenly sample up to max_items indices from range(n)."""
    if max_items <= 0 or n <= max_items:
        return list(range(n))
    step = n / max_items
    picks = []
    for i in range(max_items):
        idx = int(round(i * step))
        if idx >= n:
            idx = n - 1
        if picks and idx <= picks[-1] and picks[-1] < n - 1:
            idx = picks[-1] + 1
        picks.append(idx)
    return sorted(set(min(p, n - 1) for p in picks))
