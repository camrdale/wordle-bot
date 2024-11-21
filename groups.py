from scoring import AbstractScoringBot


class LargestRemainingBot(AbstractScoringBot):
    """A Wordle bot that minimizes the largest remaining group after a guess."""

    def __init__(self) -> None:
        super().__init__(reverse=True)

    def score(self, word: str,  remaining_solutions: frozenset[str]) -> float:
        return max([
            len(matches.intersection(remaining_solutions)) 
            for matches in self.pattern_dict[word].values()])


class MoreGroupsBot(AbstractScoringBot):
    """A Wordle bot that maximizes the number of remaining groups after a guess."""

    def score(self, word: str,  remaining_solutions: frozenset[str]) -> float:
        return len([
            1 for matches in self.pattern_dict[word].values()
            if len(matches.intersection(remaining_solutions)) > 0])
