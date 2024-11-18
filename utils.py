from collections import Counter


def calculate_pattern(guess: str, solution: str) -> tuple[int, ...]:
    """Generate a pattern list that Wordle would return if you guessed
    `guess` and the true word is `solution`
    Thanks to MarkCBell, Jeroen van der Hooft and gbotev1
    >>> calculate_pattern('weary', 'crane')
    (0, 1, 2, 1, 0)
    >>> calculate_pattern('meets', 'weary')
    (0, 2, 0, 0, 0)
    >>> calculate_pattern('rower', 'goner')
    (0, 2, 0, 2, 2)
    """
    wrong: list[int] = [i for i, v in enumerate(guess) if v != solution[i]]
    counts = Counter(solution[i] for i in wrong)
    pattern: list[int] = [2] * 5
    for i in wrong:
        v: str = guess[i]
        if counts[v] > 0:
            pattern[i] = 1
            counts[v] -= 1
        else:
            pattern[i] = 0
    return tuple(pattern)
