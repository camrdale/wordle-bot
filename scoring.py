import functools
from abc import abstractmethod

from interfaces import Bot, Game

N_GUESSES = 10
CACHE_PATTERN_DEPTH = 3
VERBOSE = False


class AbstractScoringBot(Bot):
    """A base class for Wordle bots that calculate scores for each possible guess."""

    def __init__(self, reverse: bool=False) -> None:
        super().__init__()
        self.reverse = reverse

    @abstractmethod
    def score(self, word: str,  remaining_solutions: set[str]) -> float:
        """Calculate the score for a particular guess word."""
        pass

    def __calculate_scores(
            self,
            words: list[str],
            remaining_solutions: set[str]
            ) -> list[tuple[str, float]]:
        """Calculate the score for every word in `words`."""
        if VERBOSE: print('Calculating scores for ', len(words), ' words')
        return [(word, self.score(word, remaining_solutions)) for word in words]

    def initialize(
            self,
            dictionary: list[str],
            possible_solutions: list[str],
            pattern_dict: dict[str, dict[tuple[int, ...], set[str]]]
            ) -> None:
        super().initialize(dictionary, possible_solutions, pattern_dict)
        self.dictionary = dictionary
        self.possible_solutions = possible_solutions
        self.pattern_dict: dict[str, dict[tuple[int, ...], set[str]]] = pattern_dict

        # Cache of sequence of patterns and the resulting guess.
        self.cache: dict[tuple[tuple[int, ...], ...], str] = {}

    def solve(self, game: Game) -> str | None:
        infos: list[tuple[int, ...]] = []
        candidates = list(self.dictionary)
        remaining_solutions = set(self.possible_solutions)

        for n_round in range(N_GUESSES):
            candidates = [word for word in candidates if game.hard_mode_filter(word)]
            if VERBOSE: print('Round: ', n_round + 1, '  Remaining words: ', len(remaining_solutions))
            cache_hit = False
            if len(remaining_solutions) <= 2:
                # One or two words remaining, choose the one, or pick one (can't do any better).
                guess_word = sorted(remaining_solutions)[0]
            else:
                if tuple(infos) in self.cache:
                    if VERBOSE: print('Using previously found guess')
                    guess_word = self.cache[tuple(infos)]
                    cache_hit = True
                else:
                    scores = self.__calculate_scores(candidates, remaining_solutions)

                    # Guess the candiate with highest score (lowest if `reverse`), preferring possible solutions to break ties
                    def sorting(first: tuple[str, float], second: tuple[str, float]):
                        if first[1] < second[1]:
                            return -1 if not self.reverse else 1
                        if first[1] > second[1]:
                            return 1 if not self.reverse else -1
                        if first[0] in remaining_solutions and second[0] not in remaining_solutions:
                            return 1
                        if second[0] in remaining_solutions and first[0] not in remaining_solutions:
                            return -1
                        # Fall back to sorting by word so it is deterministic.
                        if first[0] < second[0]:
                            return -1
                        return 1
                    scores.sort(key=functools.cmp_to_key(sorting), reverse=True)
                    if VERBOSE: print('Top guesses: ', [
                        (word, '{:.3f}'.format(score)) for word, score in scores[:6]])
                    guess_word = scores[0][0]
                    if n_round < CACHE_PATTERN_DEPTH:
                        self.cache[tuple(infos)] = guess_word
            info = game.guess(guess_word, cache_hit=cache_hit)
            infos.append(info)

            # Print round information
            if VERBOSE: print('Guessing: ', guess_word, '  Info: ', info)
            if info == (2, 2, 2, 2, 2):
                if VERBOSE: print(f'WIN IN {n_round + 1} GUESSES!\n\n\n')
                return guess_word

            # Filter our list of remaining possible words
            matches = self.pattern_dict[guess_word][info]
            remaining_solutions = remaining_solutions.intersection(matches)

        return None
