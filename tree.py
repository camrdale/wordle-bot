#!/usr/bin/python

import pickle
import functools
import itertools
from typing import Any, Callable, cast
from collections.abc import Iterable
from pathlib import Path
from abc import ABC, abstractmethod

import cachetools
from tqdm import tqdm
from scipy.stats import entropy # type: ignore

from interfaces import Bot, Game, CancellationWatcher
from utils import format_list

N_GUESSES = 10
VERBOSE = False
NUM_PROGRESS_BARS = 3
TREE_DIRECTORY = Path('.') / 'trees'
TREE_FILE = TREE_DIRECTORY / 'tree.p'
SAVE_TIME = False
MAX_CACHE_SIZE = 2000000


class Guess:
    """A node in the tree, representing a guess.
    
    The guess has a list of the possible patterns (edges) that can result from
    this guess, which each have an optimal guess associated with them, and
    are the nodes directly below this one in the tree. Leaf nodes are guesses
    that have no patterns (edges) connecting out of them.
    """

    def __init__(
            self,
            word: str, 
            patterns: dict[tuple[int, ...], 'Guess']
            ) -> None:
        self.word = word
        self.patterns = patterns
    
    @functools.cached_property
    def height(self) -> int:
        """The number of edges (patterns) present in the longest path connecting this node to a leaf node."""
        if len(self.patterns) == 0:
            return 0
        return max(pattern.height for pattern in self.patterns.values()) + 1
    
    @functools.cached_property
    def width(self) -> int:
        """The number of nodes that share the longest path length (i.e. the width of the base of the tree)."""
        if len(self.patterns) == 0:
            return 1
        return sum([pattern.width for pattern in self.patterns.values() if pattern.height == self.height - 1])
    
    @functools.cached_property
    def total_nodes(self) -> int:
        """The total number of nodes in the tree with this node as the root."""
        return 1 + sum(pattern.total_nodes for pattern in self.patterns.values())
    
    @functools.cached_property
    def total_guesses(self) -> int:
        """The total number of guesses it takes to exhaust the tree with this node as the root."""
        return self.total_nodes + sum(pattern.total_guesses for pattern in self.patterns.values())

    def __str__(self) -> str:
        return 'Guess({0.word}) height={0.height},width={0.width},guesses={0.total_guesses}'.format(self)
    
    def words_at_depth(self, depth: int) -> list[str]:
        """Return the words in the tree that are at the given depth."""
        if depth == 0:
            return [self.word]
        else:
            return list(itertools.chain.from_iterable(
                pattern.words_at_depth(depth-1) for pattern in self.patterns.values()))


class Optimizer(ABC):
    def __init__(
            self, 
            cancellation_watcher: CancellationWatcher,
            pattern_dict: dict[str, dict[tuple[int, ...], frozenset[str]]],
            tree_file_func: Callable[[str], Path] | None
            ) -> None:
        self.cancellation_watcher = cancellation_watcher
        self.pattern_dict = pattern_dict
        self.tree_file_func = tree_file_func

    def build_tree(
            self,
            word: str, 
            remaining_solutions: frozenset[str],
            height_limit: int,
            progress_bar_num: int
            ) -> Guess | None:
        patterns: dict[tuple[int, ...], Guess] = {}
        for pattern, matches in tqdm(self.pattern_matches(word),
                                     disable=(progress_bar_num>=NUM_PROGRESS_BARS), position=progress_bar_num, desc=word, leave=None):
            next_solutions = remaining_solutions.intersection(matches)
            if len(next_solutions) > 0 and pattern != (2, 2, 2, 2, 2):
                best_guess = self.optimal_guess(
                    next_solutions, height_limit - 1, pattern, progress_bar_num + 1)
                if best_guess is None:
                    return None
                patterns[pattern] = best_guess
        guess = Guess(word, patterns)
        if guess.height >= height_limit:
            return None
        return guess

    def load_guess(
            self, 
            word: str, 
            remaining_solutions: frozenset[str],
            height_limit: int,
            progress_bar_num: int,
            root_node: bool
            ) -> Guess | None:
        guess: Guess | None = None
        if root_node and self.tree_file_func is not None and self.tree_file_func(word).exists():
            guess = pickle.load(self.tree_file_func(word).open('rb'))
        else:
            guess = self.build_tree(word, remaining_solutions, height_limit, progress_bar_num)
            if root_node and self.tree_file_func is not None:
                pickle.dump(guess, self.tree_file_func(word).open('wb+'))
        if VERBOSE and root_node: print(
            guess if guess is not None else 'Failed({})'.format(word), ":",
            format_list(guess.words_at_depth(guess.height) if guess is not None else []),
            self.status())
        return guess

    @abstractmethod
    def optimal_guess(
            self,
            remaining_solutions: frozenset[str],
            height_limit: int,
            pattern: tuple[int, ...],
            progress_bar_num: int,
            root_node: bool=False
            ) -> Guess | None:
        pass

    def sort(
            self,
            words: Iterable[str],
            remaining_solutions: frozenset[str]
            ) -> Iterable[str]:
        """Sort words by their entropy, highest first."""
        entropy_words = [
            (cast(float, entropy([len(matches.intersection(remaining_solutions))
                                  for matches in self.pattern_dict[word].values()])),
            word) for word in words]
        return [word for _, word in sorted(entropy_words, reverse=True)]

    def status(self) -> str:
        return ''

    def pattern_matches(self, word: str) -> Iterable[tuple[tuple[int, ...], frozenset[str]]]:
        return self.pattern_dict[word].items()


class TreeBot(Bot):
    """A Wordle bot that creates an optimal decision tree of guesses for each possible pattern."""
    def __init__(
            self, 
            cancellation_watcher: CancellationWatcher,
            save_trees: bool=True,
            optimal_tree_file: Path=TREE_FILE,
            starting_word: str | None = None
            ) -> None:
        super().__init__(cancellation_watcher)
        self.save_trees = save_trees
        self.optimal_tree_file = optimal_tree_file
        self.starting_word = starting_word
        
    def initialize_with_optimizer(
            self,
            optimizer: Optimizer,
            dictionary: list[str],
            possible_solutions: frozenset[str]
            ) -> None:
        TREE_DIRECTORY.mkdir(exist_ok=True)
        self.dictionary = dictionary
        self.possible_solutions = possible_solutions
        self.tree: Guess | None = None

        if self.save_trees and self.optimal_tree_file.exists():
            print(type(self).__name__, 'Loading tree from file')
            self.tree = pickle.load(self.optimal_tree_file.open('rb'))
        else:
            print(type(self).__name__, 'Building tree of all possible guesses')
            remaining_solutions = possible_solutions
            if self.starting_word is not None:
                self.tree = optimizer.build_tree(self.starting_word, remaining_solutions, 6, 0)
                if self.tree is None:
                    print(type(self).__name__, 'ERROR: Starting word', self.starting_word, 'does not meet the given constraints.')
                    return
            else:
                self.tree = optimizer.optimal_guess(remaining_solutions, 6, (), 0, root_node=True)
                if self.tree is None:
                    print(type(self).__name__, 'ERROR: Failed to find a tree with the given constraints.')
                    return
            if not self.cancellation_watcher.is_cancelled:
                print(type(self).__name__, 'Saving optimal tree to file')
                if self.save_trees:
                    pickle.dump(self.tree, self.optimal_tree_file.open('wb+'))
        print(type(self).__name__, 'Found optimal tree:', self.tree)

    def solve(self, game: Game) -> str | None:
        if self.tree is None:
            return None

        guess = self.tree
        for n_round in range(N_GUESSES):
            pattern = game.guess(guess.word)

            # Print round information
            if VERBOSE: print(type(self).__name__, 'Guessing: ', guess.word, '  Info: ', pattern)
            if pattern == (2, 2, 2, 2, 2):
                if VERBOSE: print(type(self).__name__, f'WIN IN {n_round + 1} GUESSES!')
                return guess.word

            # Go to the next guess in the tree
            if pattern not in guess.patterns:
                if VERBOSE: print(type(self).__name__, 'Failed to find pattern', pattern, 'as possible result after guess:', guess.word)
                return None
            guess: Guess = guess.patterns[pattern]

        return None


def remaining_solutions_hashkey(self: Any, remaining_solutions: frozenset[str],  *args: Any, **kwargs: Any) -> tuple[frozenset[str]]:
    """Cache key based only on the remaining_solutions in the call (the others don't change the result)."""
    return cachetools.keys.hashkey(remaining_solutions) # type: ignore


class HeightOptimizer(Optimizer):
    """Optimizes the tree for minimizing the height and width of the base."""
    def __init__(
            self, 
            cancellation_watcher: CancellationWatcher,
            pattern_dict: dict[str, dict[tuple[int, ...], frozenset[str]]],
            tree_file_func: Callable[[str], Path] | None
            ) -> None:
        super().__init__(cancellation_watcher, pattern_dict, tree_file_func)
        self.cache: cachetools.LRUCache[tuple[frozenset[str]], Guess] = cachetools.LRUCache(maxsize=MAX_CACHE_SIZE)

    @cachetools.cachedmethod(lambda self: self.cache, key=remaining_solutions_hashkey)
    def optimal_guess(
            self,
            remaining_solutions: frozenset[str],
            height_limit: int,
            pattern: tuple[int, ...],
            progress_bar_num: int,
            root_node: bool=False
            ) -> Guess | None:
        best_guess = None
        for word in tqdm(self.sort(remaining_solutions, remaining_solutions),
                         disable=(progress_bar_num>=NUM_PROGRESS_BARS), position=progress_bar_num, leave=None,
                         desc=''.join(str(i) for i in pattern)):
            if root_node and self.cancellation_watcher.is_cancelled:
                break
            guess = self.load_guess(word, remaining_solutions, N_GUESSES, progress_bar_num + 1, root_node)
            if guess is None:
                continue
            if best_guess is None or guess.height < best_guess.height:
                best_guess = guess
            elif guess.height == best_guess.height and guess.width < best_guess.width:
                best_guess = guess
        return best_guess
    
    def status(self) -> str:
        return '(cache={:.1%})'.format(self.cache.currsize / MAX_CACHE_SIZE)


class HTreeBot(TreeBot):
    """A tree optimized for minimizing the height and width of the base."""
    def __init__(
            self, 
            cancellation_watcher: CancellationWatcher,
            save_trees: bool=True
            ) -> None:
        super().__init__(
            cancellation_watcher,
            save_trees=save_trees, 
            starting_word='slope' if SAVE_TIME else None)
        self.tree_file_func: Callable[[str], Path] | None = None
        if save_trees:
            self.tree_file_func = lambda word: TREE_DIRECTORY / (word + '_tree.p')

    def initialize(
            self,
            dictionary: list[str],
            possible_solutions: frozenset[str],
            pattern_dict: dict[str, dict[tuple[int, ...], frozenset[str]]]
            ) -> None:
        super().initialize(dictionary, possible_solutions, pattern_dict)
        optimizer = HeightOptimizer(self.cancellation_watcher, pattern_dict, self.tree_file_func)
        super().initialize_with_optimizer(optimizer, dictionary, possible_solutions)


class AverageDepthOptimizer(Optimizer):
    def __init__(
            self, 
            cancellation_watcher: CancellationWatcher,
            pattern_dict: dict[str, dict[tuple[int, ...], frozenset[str]]],
            tree_file_func: Callable[[str], Path] | None
            ) -> None:
        super().__init__(cancellation_watcher, pattern_dict, tree_file_func)
        self.cache: cachetools.LRUCache[tuple[frozenset[str]], Guess] = cachetools.LRUCache(maxsize=MAX_CACHE_SIZE)

    @cachetools.cachedmethod(lambda self: self.cache, key=remaining_solutions_hashkey)
    def optimal_guess(
            self,
            remaining_solutions: frozenset[str],
            height_limit: int,
            pattern: tuple[int, ...],
            progress_bar_num: int,
            root_node: bool=False
            ) -> Guess | None:
        best_guess = None
        for word in tqdm(self.sort(remaining_solutions, remaining_solutions),
                         disable=(progress_bar_num>=NUM_PROGRESS_BARS), position=progress_bar_num, leave=None,
                         desc=''.join(str(i) for i in pattern)):
            if root_node and self.cancellation_watcher.is_cancelled:
                break
            guess = self.load_guess(word, remaining_solutions, N_GUESSES, progress_bar_num + 1, root_node)
            if guess is None:
                continue
            if best_guess is None or guess.total_guesses < best_guess.total_guesses:
                best_guess = guess
            elif guess.total_guesses == best_guess.total_guesses and guess.height < best_guess.height:
                best_guess = guess
            elif guess.total_guesses == best_guess.total_guesses and guess.height == best_guess.height and guess.width < best_guess.width:
                best_guess = guess
        return best_guess
    
    def status(self) -> str:
        return '(cache={:.1%})'.format(self.cache.currsize / MAX_CACHE_SIZE)


class ATreeBot(TreeBot):
    """A tree optimized for minimizing the average depth (total number of guesses)."""
    def __init__(
            self, 
            cancellation_watcher: CancellationWatcher,
            save_trees: bool=True
            ) -> None:
        super().__init__(
            cancellation_watcher,
            save_trees=save_trees, 
            optimal_tree_file=(TREE_DIRECTORY / 'atree.p'), 
            starting_word='slate' if SAVE_TIME else None)
        self.tree_file_func: Callable[[str], Path] | None = None
        if save_trees:
            self.tree_file_func = lambda word: TREE_DIRECTORY / (word + '_atree.p')

    def initialize(
            self,
            dictionary: list[str],
            possible_solutions: frozenset[str],
            pattern_dict: dict[str, dict[tuple[int, ...], frozenset[str]]]
            ) -> None:
        super().initialize(dictionary, possible_solutions, pattern_dict)
        optimizer = AverageDepthOptimizer(self.cancellation_watcher, pattern_dict, self.tree_file_func)
        super().initialize_with_optimizer(optimizer, dictionary, possible_solutions)


def remaining_solutions_and_height_limit_hashkey(self: Any, remaining_solutions: frozenset[str], height_limit: int,  *args: Any, **kwargs: Any) -> tuple[frozenset[str]]:
    """Cache key based on the remaining_solutions and height limit in the call."""
    return cachetools.keys.hashkey(remaining_solutions, height_limit) # type: ignore


class IdealOptimizer(Optimizer):
    def __init__(
            self, 
            cancellation_watcher: CancellationWatcher,
            pattern_dict: dict[str, dict[tuple[int, ...], frozenset[str]]],
            tree_file_func: Callable[[str], Path] | None
            ) -> None:
        super().__init__(cancellation_watcher, pattern_dict, tree_file_func)
        self.cache: cachetools.LRUCache[tuple[frozenset[str]], Guess] = cachetools.LRUCache(maxsize=MAX_CACHE_SIZE)

    @cachetools.cachedmethod(lambda self: self.cache, key=remaining_solutions_and_height_limit_hashkey)
    def optimal_guess(
            self,
            remaining_solutions: frozenset[str],
            height_limit: int,
            pattern: tuple[int, ...],
            progress_bar_num: int,
            root_node: bool=False
            ) -> Guess | None:
        best_guess = None
        for word in tqdm(self.sort(remaining_solutions, remaining_solutions),
                         disable=(progress_bar_num>=NUM_PROGRESS_BARS), position=progress_bar_num, leave=None,
                         desc=''.join(str(i) for i in pattern)):
            if root_node and self.cancellation_watcher.is_cancelled:
                break
            guess = self.load_guess(word, remaining_solutions, height_limit, progress_bar_num + 1, root_node)
            if guess is None:
                continue
            if best_guess is None or guess.total_guesses < best_guess.total_guesses:
                best_guess = guess
        return best_guess
    
    def status(self) -> str:
        return '(cache={:.1%})'.format(self.cache.currsize / MAX_CACHE_SIZE)


class ITreeBot(TreeBot):
    """A tree optimized for first not failing any words, then to minimize the average depth."""
    def __init__(
            self, 
            cancellation_watcher: CancellationWatcher,
            save_trees: bool=True
            ) -> None:
        super().__init__(
            cancellation_watcher,
            save_trees=save_trees, 
            optimal_tree_file=(TREE_DIRECTORY / 'itree.p'),
            starting_word='slope' if SAVE_TIME else None)
        self.tree_file_func: Callable[[str], Path] | None = None
        if save_trees:
            self.tree_file_func = lambda word: TREE_DIRECTORY / (word + '_itree.p')

    def initialize(
            self,
            dictionary: list[str],
            possible_solutions: frozenset[str],
            pattern_dict: dict[str, dict[tuple[int, ...], frozenset[str]]]
            ) -> None:
        super().initialize(dictionary, possible_solutions, pattern_dict)
        optimizer = IdealOptimizer(self.cancellation_watcher, pattern_dict, self.tree_file_func)
        super().initialize_with_optimizer(optimizer, dictionary, possible_solutions)


if __name__ == "__main__":
    from wordle import GameImpl, generate_pattern_dict
    possible_solutions = frozenset(['crane', 'bears', 'weary'])
    dictionary = list(possible_solutions)
    pattern_dict = generate_pattern_dict(dictionary, possible_solutions)
    bot = ITreeBot(CancellationWatcher(), save_trees=False)
    bot.initialize(dictionary, possible_solutions, pattern_dict)
    for word_to_guess in possible_solutions:
        game = GameImpl(word_to_guess, True)
        result = bot.solve(game)
        print('For', word_to_guess, 'took', game.num_guesses(), 'to get:', result)
