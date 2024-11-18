import os
import functools
import pickle
from tqdm import tqdm
from scipy.stats import entropy # type: ignore
from collections import defaultdict

from interfaces import Bot, Game
from utils import calculate_pattern

N_GUESSES = 10
DB_FILE = 'pattern_dict.p'
SAVE_TIME = True
VERBOSE = False


class EntropyBot(Bot):
    """A Wordle bot that uses entropy of guesses to choose the words to guess."""

    def __generate_pattern_dict(self,
                                dictionary: list[str],
                                possible_solutions: list[str]
                                ) -> dict[str, dict[tuple[int, ...], set[str]]]:
        """For each word and possible information returned, store a list
        of candidate words
        >>> pattern_dict = generate_pattern_dict(['weary', 'bears', 'crane'])
        >>> pattern_dict['crane'][(2, 2, 2, 2, 2)]
        {'crane'}
        >>> sorted(pattern_dict['crane'][(0, 1, 2, 0, 1)])
        ['bears', 'weary']
        """
        pattern_dict: dict[str, dict[tuple[int, ...], set[str]]] = defaultdict(lambda: defaultdict(set))
        for word in tqdm(dictionary):
            for word2 in possible_solutions:
                pattern = calculate_pattern(word, word2)
                pattern_dict[word][pattern].add(word2)
        return dict(pattern_dict)


    def __calculate_entropies(self,
                              words: list[str],
                              remaining_solutions: set[str]
                              ) -> list[tuple[str, float]]:
        """Calculate the entropy for every word in `words`, taking into account
        the `remaining_solutions`"""
        if VERBOSE: print('Calculating entropies for ', len(words), ' words')
        entropies: list[tuple[str, float]] = []
        for word in words:
            entropies.append((word, entropy(
                [len(matches.intersection(remaining_solutions)) 
                for matches in self.pattern_dict[word].values()]))) # type: ignore
        return entropies

    def initialize(self, dictionary: list[str], possible_solutions: list[str]) -> None:
        super().initialize(dictionary, possible_solutions)
        self.dictionary = dictionary
        self.possible_solutions = possible_solutions

        # Calculate the pattern_dict and cache it, or load the cache.
        if DB_FILE in os.listdir('.'):
            print('Loading pattern dictionary from file')
            self.pattern_dict: dict[str, dict[tuple[int, ...], set[str]]] = pickle.load(open(DB_FILE, 'rb'))
        else:
            print('Generating pattern dictionary')
            self.pattern_dict = self.__generate_pattern_dict(dictionary, possible_solutions)
            print('Saving pattern dictionary to file')
            pickle.dump(self.pattern_dict, open(DB_FILE, 'wb+'))

        # Cache of first round info's and the resulting guess.
        self.initial_cache: dict[tuple[int, ...], str] = {}

    def solve(self, game: Game) -> str | None:
        info: tuple[int, ...] = ()
        candidates = list(self.dictionary)

        if SAVE_TIME:
            guess_word = 'soare'
            remaining_solutions = set(self.possible_solutions)
            info = game.guess(guess_word)
            if VERBOSE: print('Guessing: ', guess_word, '  Info: ', info)
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
                  entropies = self.__calculate_entropies(candidates, remaining_solutions)

                  # Guess the candiate with highest entropy, preferring possible solutions to break ties
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
                  entropies.sort(key=functools.cmp_to_key(sorting), reverse=True)
                  if VERBOSE: print('Top guesses: ', [(word, '{:.5f}'.format(entropy))
                                          for word, entropy in entropies[:6]])
                  guess_word = entropies[0][0]
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
