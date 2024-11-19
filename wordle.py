#!/usr/bin/python
import os
import pickle
import signal
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import Any

from interfaces import Bot, Game
from utils import calculate_pattern
from entropy import EntropyBot
from groups import LargestRemainingBot, MoreGroupsBot
from rando import RandomBot

DICT_FILE = 'all_words.txt'
SOLUTIONS_FILE = 'words.txt'
DB_FILE = 'pattern_dict.p'
HARD_MODE = True


class GameImpl(Game):
    def __init__(self, solution: str, hard_mode: bool) -> None:
        self.__solution = solution
        self.__hard_mode = hard_mode
        
        self.__guesses: list[tuple[str, tuple[int, ...]]] = []

    def guess(self, word: str) -> tuple[int, ...]:
        pattern = calculate_pattern(word, self.__solution)
        self.__guesses.append((word, pattern))
        return pattern

    def hard_mode_filter(self, word: str) -> bool:
        if not self.__hard_mode:
            return True
        for guess_word, info in self.__guesses:
            for i, c in enumerate(guess_word):
                if info[i] == 2 and word[i] != c:
                    return False
            counts = Counter(c for c, result in zip(guess_word, info) if result > 0)
            counts.subtract(word)
            if len(list(counts.elements())) > 0:
                return False
        return True
    
    def num_guesses(self) -> int:
        return len(self.__guesses)


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


def format_list(words: list[str]) -> str:
    output = words[:10]
    if len(words) > 10:
        output.append('...')
    return ', '.format(output)


abort = False

def main():
    # load all 5-letter-words for making patterns 
    with open(DICT_FILE) as ifp:
        dictionary = list(map(lambda x: x.strip(), ifp.readlines()))

    # Load 2315 words for solutions
    with open(SOLUTIONS_FILE) as ifp:
        possible_solutions = list(map(lambda x: x.strip(), ifp.readlines()))

    # Calculate the pattern_dict and cache it, or load the cache.
    if DB_FILE in os.listdir('.'):
        print('Loading pattern dictionary from file')
        pattern_dict: dict[str, dict[tuple[int, ...], set[str]]] = pickle.load(open(DB_FILE, 'rb'))
    else:
        print('Generating pattern dictionary')
        pattern_dict = generate_pattern_dict(dictionary, possible_solutions)
        print('Saving pattern dictionary to file')
        pickle.dump(pattern_dict, open(DB_FILE, 'wb+'))

    # Don't quit immediately on Ctrl-C, finish the iteratation and print the results.
    def stop(signum: int, frame: Any):
        print('Aborting after this iteration')
        global abort
        abort = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    # Overall accumulated stats across all bots and words.
    stats: dict[str, dict[str, list[Any]]] = {}

    bots: dict[str, Bot] = {
        "g-more": MoreGroupsBot(),
        "g-small": LargestRemainingBot(),
        "entropy": EntropyBot(),
        "rando": RandomBot(),
        }
    for bot_name, bot in bots.items():
        if abort:
            break

        stats[bot_name] = defaultdict(list)
        bot.initialize(dictionary, possible_solutions, pattern_dict)

        print('Starting to solve with', bot_name)
        for word_to_guess in tqdm(possible_solutions):
            if abort:
                break
            
            game = GameImpl(word_to_guess, HARD_MODE)
            result = bot.solve(game)

            if result is None:
                stats[bot_name]['failed'].append(word_to_guess)
            elif result != word_to_guess:
                stats[bot_name]['wrong'].append(word_to_guess)
            else:
                num_guesses = game.num_guesses()
                stats[bot_name]['guesses'].append(num_guesses)
                if num_guesses > 6:
                    stats[bot_name]['misses'].append(word_to_guess)

    print('Bots:\t\t\t', "\t".join(stats.keys()))
    print('Successful guesses:\t', "\t".join(str(len(bot_stats['guesses'])) for bot_stats in stats.values()))
    print('Average guess count:\t', "\t".join('{:.3f}'.format(1.0 * sum(bot_stats['guesses']) / len(bot_stats['guesses'])) for bot_stats in stats.values()))
    print('Guess counts of\t  1:\t', "\t".join(str(len([x for x in bot_stats['guesses'] if x == 1])) for bot_stats in stats.values()))
    print('\t\t  2:\t', "\t".join(str(len([x for x in bot_stats['guesses'] if x == 2])) for bot_stats in stats.values()))
    print('\t\t  3:\t', "\t".join(str(len([x for x in bot_stats['guesses'] if x == 3])) for bot_stats in stats.values()))
    print('\t\t  4:\t', "\t".join(str(len([x for x in bot_stats['guesses'] if x == 4])) for bot_stats in stats.values())) 
    print('\t\t  5:\t', "\t".join(str(len([x for x in bot_stats['guesses'] if x == 5])) for bot_stats in stats.values())) 
    print('\t\t  6:\t', "\t".join(str(len([x for x in bot_stats['guesses'] if x == 6])) for bot_stats in stats.values()))
    print('\t\t 7+:\t', "\t".join(str(len([x for x in bot_stats['guesses'] if x > 6])) for bot_stats in stats.values()))
    for bot_name, bot_stats in stats.items():
        if 'misses' in bot_stats:
            print('Bot', bot_name, 'missed words:', format_list(bot_stats['misses']))
        if 'wrong' in bot_stats:
            print('Bot', bot_name, 'wrong answers:', format_list(bot_stats['wrong']))
        if 'failed' in bot_stats:
            print('Bot', bot_name, 'failed to complete:', format_list(bot_stats['failed']))


if __name__ == "__main__":
    main()
