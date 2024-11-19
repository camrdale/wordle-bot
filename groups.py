import functools

from interfaces import Bot, Game

N_GUESSES = 10
SAVE_TIME = True
VERBOSE = False


class LargestRemainingBot(Bot):
    """A Wordle bot that minimizes the largest remaining group after a guess."""

    def __calculate_largest_remaining(self,
                                      words: list[str],
                                      remaining_solutions: set[str]
                                      ) -> list[tuple[str, int]]:
        """Calculate the largest remaining group for each possible guess."""
        if VERBOSE: print('Calculating largest remaining groups for ', len(words), ' words')
        largest_remaining: list[tuple[str, int]] = []
        for word in words:
            largest_remaining.append((word, max(
                [len(matches.intersection(remaining_solutions)) 
                 for matches in self.pattern_dict[word].values()])))
        return largest_remaining

    def initialize(self,
                   dictionary: list[str],
                   possible_solutions: list[str],
                   pattern_dict: dict[str, dict[tuple[int, ...], set[str]]]
                   ) -> None:
        super().initialize(dictionary, possible_solutions, pattern_dict)
        self.dictionary = dictionary
        self.possible_solutions = possible_solutions
        self.pattern_dict: dict[str, dict[tuple[int, ...], set[str]]] = pattern_dict

        # Cache of first round info's and the resulting guess.
        self.initial_cache: dict[tuple[int, ...], str] = {}

    def solve(self, game: Game) -> str | None:
        info: tuple[int, ...] = ()
        candidates = list(self.dictionary)

        if SAVE_TIME:
            guess_word = 'raise'
            remaining_solutions = set(self.possible_solutions)
            info = game.guess(guess_word)
            if VERBOSE: print('Guessing: ', guess_word, '  Info: ', info)
            if info == (2, 2, 2, 2, 2):
                if VERBOSE: print(f'WIN IN 1 GUESSES!\n\n\n')
                return guess_word
            matches = self.pattern_dict[guess_word][info]
            remaining_solutions = remaining_solutions.intersection(matches)
            init_round = 1
        else:
            remaining_solutions = set(self.possible_solutions)
            init_round = 0

        for n_round in range(init_round, N_GUESSES):
            candidates = [word for word in candidates if game.hard_mode_filter(word)]
            if VERBOSE: print('Round: ', n_round + 1, '  Remaining words: ', len(remaining_solutions))
            if len(remaining_solutions) <= 2:
                # One or two words remaining, choose the one, or pick one (can't do any better).
                guess_word = sorted(remaining_solutions)[0]
            else:
                if n_round == 1 and info in self.initial_cache:
                  if VERBOSE: print('Using previously found guess')
                  guess_word = self.initial_cache[info]
                else:
                  largest_remaining = self.__calculate_largest_remaining(candidates, remaining_solutions)

                  # Guess the candiate which minimizes remaining group sizes, preferring possible solutions to break ties
                  def sorting(first: tuple[str, float], second: tuple[str, float]):
                      if first[1] > second[1]:
                          return -1
                      if first[1] < second[1]:
                          return 1
                      if first[0] in remaining_solutions and second[0] not in remaining_solutions:
                          return 1
                      if second[0] in remaining_solutions and first[0] not in remaining_solutions:
                          return -1
                      # Fall back to sorting by word so it is deterministic.
                      if first[0] < second[0]:
                          return -1
                      return 1
                  largest_remaining.sort(key=functools.cmp_to_key(sorting), reverse=True)
                  if VERBOSE: print('Top guesses: ', largest_remaining[:6])
                  guess_word = largest_remaining[0][0]
                  if n_round == 1:
                      self.initial_cache[info] = guess_word
            info = game.guess(guess_word)

            # Print round information
            if VERBOSE: print('Guessing: ', guess_word, '  Info: ', info)
            if info == (2, 2, 2, 2, 2):
                if VERBOSE: print(f'WIN IN {n_round + 1} GUESSES!\n\n\n')
                return guess_word

            # Filter our list of remaining possible words
            matches = self.pattern_dict[guess_word][info]
            remaining_solutions = remaining_solutions.intersection(matches)

        return None


class MoreGroupsBot(Bot):
    """A Wordle bot that maximizes the number of remaining groups after a guess."""

    def __calculate_remaining_groups(self,
                                     words: list[str],
                                     remaining_solutions: set[str]
                                     ) -> list[tuple[str, int]]:
        """Calculate the number of remaining groups for each possible guess."""
        if VERBOSE: print('Calculating number of remaining groups for ', len(words), ' words')
        remaining_groups: list[tuple[str, int]] = []
        for word in words:
            remaining_groups.append((word, len(
                [1 for matches in self.pattern_dict[word].values()
                 if len(matches.intersection(remaining_solutions)) > 0])))
        return remaining_groups

    def initialize(self,
                   dictionary: list[str],
                   possible_solutions: list[str],
                   pattern_dict: dict[str, dict[tuple[int, ...], set[str]]]
                   ) -> None:
        super().initialize(dictionary, possible_solutions, pattern_dict)
        self.dictionary = dictionary
        self.possible_solutions = possible_solutions
        self.pattern_dict: dict[str, dict[tuple[int, ...], set[str]]] = pattern_dict

        # Cache of first round info's and the resulting guess.
        self.initial_cache: dict[tuple[int, ...], str] = {}

    def solve(self, game: Game) -> str | None:
        info: tuple[int, ...] = ()
        candidates = list(self.dictionary)

        if SAVE_TIME:
            guess_word = 'trace'
            remaining_solutions = set(self.possible_solutions)
            info = game.guess(guess_word)
            if VERBOSE: print('Guessing: ', guess_word, '  Info: ', info)
            if info == (2, 2, 2, 2, 2):
                if VERBOSE: print(f'WIN IN 1 GUESSES!\n\n\n')
                return guess_word
            matches = self.pattern_dict[guess_word][info]
            remaining_solutions = remaining_solutions.intersection(matches)
            init_round = 1
        else:
            remaining_solutions = set(self.possible_solutions)
            init_round = 0

        for n_round in range(init_round, N_GUESSES):
            candidates = [word for word in candidates if game.hard_mode_filter(word)]
            if VERBOSE: print('Round: ', n_round + 1, '  Remaining words: ', len(remaining_solutions))
            if len(remaining_solutions) <= 2:
                # One or two words remaining, choose the one, or pick one (can't do any better).
                guess_word = sorted(remaining_solutions)[0]
            else:
                if n_round == 1 and info in self.initial_cache:
                  if VERBOSE: print('Using previously found guess')
                  guess_word = self.initial_cache[info]
                else:
                  remaining_groups = self.__calculate_remaining_groups(candidates, remaining_solutions)

                  # Guess the candiate which maximizes the number of remaining groups, preferring possible solutions to break ties
                  def sorting(first: tuple[str, float], second: tuple[str, float]):
                      if first[1] < second[1]:
                          return -1
                      if first[1] > second[1]:
                          return 1
                      if first[0] in remaining_solutions and second[0] not in remaining_solutions:
                          return 1
                      if second[0] in remaining_solutions and first[0] not in remaining_solutions:
                          return -1
                      # Fall back to sorting by word so it is deterministic.
                      if first[0] < second[0]:
                          return -1
                      return 1
                  remaining_groups.sort(key=functools.cmp_to_key(sorting), reverse=True)
                  if VERBOSE: print('Top guesses: ', remaining_groups[:60])
                  guess_word = remaining_groups[0][0]
                  if n_round == 1:
                      self.initial_cache[info] = guess_word
            info = game.guess(guess_word)

            # Print round information
            if VERBOSE: print('Guessing: ', guess_word, '  Info: ', info)
            if info == (2, 2, 2, 2, 2):
                if VERBOSE: print(f'WIN IN {n_round + 1} GUESSES!\n\n\n')
                return guess_word

            # Filter our list of remaining possible words
            matches = self.pattern_dict[guess_word][info]
            remaining_solutions = remaining_solutions.intersection(matches)

        return None
