#!/usr/bin/python
import signal
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import Any, Optional

from interfaces import Bot, Game
from utils import calculate_pattern
from entropy import EntropyBot

DICT_FILE = 'all_words.txt'
SOLUTIONS_FILE = 'words.txt'
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


abort = False

def main():
    # load all 5-letter-words for making patterns 
    with open(DICT_FILE) as ifp:
        dictionary = list(map(lambda x: x.strip(), ifp.readlines()))

    # Load 2315 words for solutions
    with open(SOLUTIONS_FILE) as ifp:
        possible_solutions = list(map(lambda x: x.strip(), ifp.readlines()))

    bot: Bot = EntropyBot()
    bot.initialize(dictionary, possible_solutions)

    # Overall accumulated stats across all words.
    stats: dict[str, list[str | int]] = defaultdict(list)

    # Don't quit immediately on Ctrl-C, finish the iteratation and print the results.
    def stop(signum: int, frame: Optional[Any]):
        print('Aborting after this iteration')
        global abort
        abort = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print('Starting to solve')
    for word_to_guess in tqdm(possible_solutions):
        if abort:
            break
        
        game = GameImpl(word_to_guess, HARD_MODE)
        result = bot.solve(game)

        if result is None:
            stats['failed'].append(word_to_guess)
        elif result != word_to_guess:
            stats['wrong'].append(word_to_guess)
        else:
            num_guesses = game.num_guesses()
            stats['guesses'].append(num_guesses)
            if num_guesses > 6:
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
    print('Wrong answers:', stats['wrong'])
    print('Failed to complete:', stats['failed'])


if __name__ == "__main__":
    main()
