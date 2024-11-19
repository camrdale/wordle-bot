from scipy.stats import entropy # type: ignore

from scoring import AbstractScoringBot


class EntropyBot(AbstractScoringBot):
    """A Wordle bot that uses entropy of guesses to choose the words to guess."""
    
    def score(self, word: str,  remaining_solutions: set[str]) -> float:
        return entropy([
            len(matches.intersection(remaining_solutions)) 
            for matches in self.pattern_dict[word].values()]) # type: ignore
