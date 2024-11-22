#!/usr/bin/python

import os
import pickle
import signal
import time
from datetime import timedelta
from collections import defaultdict, Counter
from typing import Any

from tqdm import tqdm

from interfaces import Bot, Game, CancellationWatcher
from utils import calculate_pattern, format_list
from entropy import EntropyBot
from groups import LargestRemainingBot, MoreGroupsBot
from tree import HTreeBot, ATreeBot, ITreeBot
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
        self.__cache_hits: dict[int, int] = defaultdict(int)

    def guess(self, word: str, cache_hit: bool=False) -> tuple[int, ...]:
        pattern = calculate_pattern(word, self.__solution)
        self.__guesses.append((word, pattern))
        if cache_hit:
            self.__cache_hits[len(self.__guesses)] += 1
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
    
    def starting_word(self) -> str:
        return self.__guesses[0][0]
    
    def cache_hits(self) -> dict[int, int]:
        return self.__cache_hits


def generate_pattern_dict(dictionary: list[str],
                          possible_solutions: frozenset[str]
                          ) -> dict[str, dict[tuple[int, ...], frozenset[str]]]:
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
        for solution in possible_solutions:
            pattern = calculate_pattern(word, solution)
            pattern_dict[word][pattern].add(solution)
    return {word: {pattern: frozenset(matches) 
                   for pattern, matches in pattern_matches.items()} 
            for word, pattern_matches in pattern_dict.items()}


def main():
    # load all 5-letter-words for making patterns 
    with open(DICT_FILE) as ifp:
        dictionary = list(map(lambda x: x.strip(), ifp.readlines()))

    # Load 2315 words for solutions
    with open(SOLUTIONS_FILE) as ifp:
        possible_solutions = frozenset(map(lambda x: x.strip(), ifp.readlines()))

    # Calculate the pattern_dict and cache it, or load the cache.
    start = time.time()
    if DB_FILE in os.listdir('.'):
        print('Loading pattern dictionary from file')
        pattern_dict: dict[str, dict[tuple[int, ...], frozenset[str]]] = pickle.load(open(DB_FILE, 'rb'))
    else:
        print('Generating pattern dictionary')
        pattern_dict = generate_pattern_dict(dictionary, possible_solutions)
        print('Saving pattern dictionary to file')
        pickle.dump(pattern_dict, open(DB_FILE, 'wb+'))
    pattern_dict_time= time.time() - start

    # Don't quit on Ctrl-C, cancel the initialize/solve and proceed to the next step.
    cancellation_watcher = CancellationWatcher()
    signal.signal(signal.SIGINT, cancellation_watcher.stop)

    # Overall accumulated stats across all bots and words.
    stats: dict[str, dict[str, Any]] = {}

    bots: dict[str, Bot] = {
        "i-tree": ITreeBot(cancellation_watcher),
        "a-tree": ATreeBot(cancellation_watcher),
        "h-tree": HTreeBot(cancellation_watcher),
        "g-more": MoreGroupsBot(cancellation_watcher),
        "g-small": LargestRemainingBot(cancellation_watcher),
        "entropy": EntropyBot(cancellation_watcher),
        "rando": RandomBot(cancellation_watcher),
        }
    for bot_name, bot in bots.items():
        stats[bot_name] = defaultdict(list)
        stats[bot_name]['cache_hits'] = Counter()
        stats[bot_name]['starting_word'] = set()
        cancellation_watcher.is_cancelled = False

        start = time.time()
        bot.initialize(dictionary, possible_solutions, pattern_dict)
        stats[bot_name]['initialize_time'] = time.time() - start + pattern_dict_time

        print('Starting to solve with', bot_name)
        cancellation_watcher.is_cancelled = False
        start = time.time()
        for word_to_guess in tqdm(possible_solutions):
            if cancellation_watcher.is_cancelled:
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
            stats[bot_name]['cache_hits'].update(game.cache_hits())
            stats[bot_name]['starting_word'].add(game.starting_word())

        stats[bot_name]['solve_time'] = time.time() - start

    print('Bots:\t\t\t', "\t".join(stats.keys()), sep='')
    print('Initialize time:\t', "\t".join(str(timedelta(seconds=int(bot_stats['initialize_time']))) for bot_stats in stats.values()), sep='')
    print('Solve time:\t\t', "\t".join(str(timedelta(seconds=int(bot_stats['solve_time']))) for bot_stats in stats.values()), sep='')
    print('Successful guesses:\t', "\t".join(str(len(bot_stats['guesses'])) for bot_stats in stats.values()), sep='')
    print('Average guess count:\t', "\t".join('{:.3f}'.format(1.0 * sum(bot_stats['guesses']) / len(bot_stats['guesses'])) for bot_stats in stats.values()), sep='')
    print('Guess counts of\t  1:\t', "\t".join(str(len([x for x in bot_stats['guesses'] if x == 1])) for bot_stats in stats.values()), sep='')
    print('\t\t  2:\t', "\t".join(str(len([x for x in bot_stats['guesses'] if x == 2])) for bot_stats in stats.values()), sep='')
    print('\t\t  3:\t', "\t".join(str(len([x for x in bot_stats['guesses'] if x == 3])) for bot_stats in stats.values()), sep='')
    print('\t\t  4:\t', "\t".join(str(len([x for x in bot_stats['guesses'] if x == 4])) for bot_stats in stats.values()), sep='')
    print('\t\t  5:\t', "\t".join(str(len([x for x in bot_stats['guesses'] if x == 5])) for bot_stats in stats.values()), sep='')
    print('\t\t  6:\t', "\t".join(str(len([x for x in bot_stats['guesses'] if x == 6])) for bot_stats in stats.values()), sep='')
    print('\t\t 7+:\t', "\t".join(str(len([x for x in bot_stats['guesses'] if x > 6])) for bot_stats in stats.values()), sep='')
    print('Starting word:\t\t', "\t".join(list(bot_stats['starting_word'])[0] if len(bot_stats['starting_word']) == 1 else '[{}]'.format(len(bot_stats['starting_word'])) for bot_stats in stats.values()), sep='')
    for bot_name, bot_stats in stats.items():
        if 'misses' in bot_stats:
            print('Bot', bot_name, 'missed words:', format_list(bot_stats['misses']))
        if 'wrong' in bot_stats:
            print('Bot', bot_name, 'wrong answers:', format_list(bot_stats['wrong']))
        if 'failed' in bot_stats:
            print('Bot', bot_name, 'failed to complete:', format_list(bot_stats['failed']))
    for bot_name, bot_stats in stats.items():
        if len(bot_stats['cache_hits']) > 0:
            print(bot_name, 'cache hits:', dict(bot_stats['cache_hits']))

if __name__ == "__main__":
    main()
