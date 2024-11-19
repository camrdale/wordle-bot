import random

from interfaces import Bot, Game

N_GUESSES = 10
VERBOSE = False


class RandomBot(Bot):
    """A terrible Wordle bot that chooses words randomly."""

    def initialize(self,
                   dictionary: list[str],
                   possible_solutions: list[str],
                   pattern_dict: dict[str, dict[tuple[int, ...], set[str]]]
                   ) -> None:
        super().initialize(dictionary, possible_solutions, pattern_dict)
        self.dictionary = dictionary
        self.possible_solutions = possible_solutions
        self.pattern_dict = pattern_dict

    def solve(self, game: Game) -> str | None:
        remaining_solutions = self.possible_solutions

        for n_round in range(0, N_GUESSES):
            if VERBOSE: print('Round: ', n_round + 1, '  Remaining words: ', len(remaining_solutions))

            guess_word = random.choice(remaining_solutions)
            info = game.guess(guess_word)

            # Print round information
            if VERBOSE: print('Guessing: ', guess_word, '  Info: ', info)
            if info == (2, 2, 2, 2, 2):
                if VERBOSE: print(f'WIN IN {n_round + 1} GUESSES!\n\n\n')
                return guess_word

            # Filter our list of remaining possible words
            remaining_solutions = [word for word in remaining_solutions if game.hard_mode_filter(word)]

        return None
