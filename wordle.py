#!/usr/bin/python
import os
import functools
import pickle
import signal
from tqdm import tqdm
from scipy.stats import entropy # type: ignore
from collections import defaultdict, Counter
from typing import Any, Optional

N_GUESSES = 10
DICT_FILE = 'all_words.txt'
SOLUTIONS_FILE = 'words.txt'
DB_FILE = 'pattern_dict.p'
SAVE_TIME = True
HARD_MODE = True
VERBOSE = False

def calculate_pattern(guess: str, true: str) -> tuple[int, ...]:
    """Generate a pattern list that Wordle would return if you guessed
    `guess` and the true word is `true`
    Thanks to MarkCBell, Jeroen van der Hooft and gbotev1
    >>> calculate_pattern('weary', 'crane')
    (0, 1, 2, 1, 0)
    >>> calculate_pattern('meets', 'weary')
    (0, 2, 0, 0, 0)
    >>> calculate_pattern('rower', 'goner')
    (0, 2, 0, 2, 2)
    """
    wrong: list[int] = [i for i, v in enumerate(guess) if v != true[i]]
    counts = Counter(true[i] for i in wrong)
    pattern: list[int] = [2] * 5
    for i in wrong:
        v: str = guess[i]
        if counts[v] > 0:
            pattern[i] = 1
            counts[v] -= 1
        else:
            pattern[i] = 0
    return tuple(pattern)


def generate_pattern_dict(dictionary: list[str],
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


def calculate_entropies(words: list[str],
                        remaining_solutions: set[str], 
                        pattern_dict: dict[str, dict[tuple[int, ...], set[str]]]
                        ) -> list[tuple[str, float]]:
    """Calculate the entropy for every word in `words`, taking into account
    the `remaining_solutions`"""
    if VERBOSE: print('Calculating entropies for ', len(words), ' words')
    entropies: list[tuple[str, float]] = []
    for word in words:
        entropies.append((word, entropy(
            [len(matches.intersection(remaining_solutions)) 
             for matches in pattern_dict[word].values()]))) # type: ignore
    return entropies


def hard_mode_filter(guesses: list[tuple[str, tuple[int, ...]]], word: str) -> bool:
    if not HARD_MODE:
        return True
    for guess_word, info in guesses:
        for i, c in enumerate(guess_word):
            if info[i] == 2 and word[i] != c:
                return False
        counts = Counter(c for c, result in zip(guess_word, info) if result > 0)
        counts.subtract(word)
        if len(list(counts.elements())) > 0:
            return False
    return True


abort = False

def main():
    # load all 5-letter-words for making patterns 
    with open(DICT_FILE) as ifp:
        all_dictionary = list(map(lambda x: x.strip(), ifp.readlines()))

    # Load 2315 words for solutions
    with open(SOLUTIONS_FILE) as ifp:
        possible_solutions = list(map(lambda x: x.strip(), ifp.readlines()))

    # Calculate the pattern_dict and cache it, or load the cache.
    if DB_FILE in os.listdir('.'):
        print('Loading pattern dictionary from file')
        pattern_dict: dict[str, dict[tuple[int, ...], set[str]]] = pickle.load(open(DB_FILE, 'rb'))
    else:
        print('Generating pattern dictionary')
        pattern_dict = generate_pattern_dict(all_dictionary, possible_solutions)
        print('Saving pattern dictionary to file')
        pickle.dump(pattern_dict, open(DB_FILE, 'wb+'))

    # Cache of first round info's and the resulting guess.
    initial_cache: dict[tuple[int, ...], str] = {}

    # Overall accumulated stats across all words.
    stats: dict[str, list[str | int]] = defaultdict(list)

    # Don't quit immediately on Ctrl-C, finish the iteratation and print the results.
    def stop(signum: int, frame: Optional[Any]):
        print('Aborting after this iteration')
        global abort
        abort = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    for word_to_guess in tqdm(possible_solutions):
        if abort:
            break

        info: tuple[int, ...] = ()
        guesses: list[tuple[str, tuple[int, ...]]] = []
        candidates = list(all_dictionary)

        if SAVE_TIME:
            guess_word = 'soare'
            remaining_solutions = set(possible_solutions)
            info = calculate_pattern(guess_word, word_to_guess)
            guesses.append((guess_word, info))
            if VERBOSE: print('Guessing: ', guess_word, '  Info: ', info)
            matches = pattern_dict[guess_word][info]
            remaining_solutions = remaining_solutions.intersection(matches)
            init_round = 1
        else:
            remaining_solutions = set(possible_solutions)
            init_round = 0

        for n_round in range(init_round, N_GUESSES):
            candidates = [word for word in candidates if hard_mode_filter(guesses, word)]
            if VERBOSE: print('Round: ', n_round + 1, '  Remaining words: ', len(remaining_solutions))
            if len(remaining_solutions) <= 2:
                # One or two words remaining, choose the one, or pick one (can't do any better).
                guess_word = sorted(remaining_solutions)[0]
            else:
                if n_round == 1 and info in initial_cache:
                  if VERBOSE: print('Using previously found guess')
                  guess_word = initial_cache[info]
                else:
                  entropies = calculate_entropies(candidates, remaining_solutions, pattern_dict)

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
                      initial_cache[info] = guess_word
            info = calculate_pattern(guess_word, word_to_guess)

            # Print round information
            if VERBOSE: print('Guessing: ', guess_word, '  Info: ', info)
            if info == (2, 2, 2, 2, 2):
                if VERBOSE: print(f'WIN IN {n_round + 1} GUESSES!\n\n\n')
                stats['guesses'].append(n_round + 1)
                if n_round + 1 > 6:
                    stats['misses'].append(word_to_guess)
                break

            guesses.append((guess_word, info))

            # Filter our list of remaining possible words
            matches = pattern_dict[guess_word][info]
            remaining_solutions = remaining_solutions.intersection(matches)
        else:
            stats['misses'].append(word_to_guess)

    print(len(stats['guesses']), 'successful guesses')
    print('Average guess count:', 1.0 * sum(stats['guesses']) / len(stats['guesses'])) # type: ignore
    print('Guess counts of')
    print('  1:', len([x for x in stats['guesses'] if x == 1]))
    print('  2:', len([x for x in stats['guesses'] if x == 2]))
    print('  3:', len([x for x in stats['guesses'] if x == 3]))
    print('  4:', len([x for x in stats['guesses'] if x == 4])) 
    print('  5:', len([x for x in stats['guesses'] if x == 5])) 
    print('  6:', len([x for x in stats['guesses'] if x == 6]))
    print(' 7+:', len([x for x in stats['guesses'] if x > 6])) # type: ignore
    print('Missed words:', stats['misses'])

if __name__ == "__main__":
    main()
