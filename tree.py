#!/usr/bin/python

import pickle
import functools
import itertools
from typing import cast
from collections.abc import Iterable
from pathlib import Path
from abc import ABC, abstractmethod

import cachetools
from tqdm import tqdm
from scipy.stats import entropy # type: ignore

from interfaces import Bot, Game
from utils import format_list

N_GUESSES = 10
VERBOSE = False
TREE_DIRECTORY = Path('.') / 'trees'
TREE_FILE = TREE_DIRECTORY / 'tree.p'
SAVE_TIME = False
MAX_CACHE_SIZE = 2000000


class Optimizer(ABC):
    def __init__(
            self, 
            pattern_dict: dict[str, dict[tuple[int, ...], frozenset[str]]]
            ) -> None:
        self.pattern_dict = pattern_dict

    @abstractmethod
    def optimal_guess(self, remaining_solutions: frozenset[str]) -> 'Guess':
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
            remaining_solutions: frozenset[str],
            optimizer: Optimizer
            ) -> None:
        self.word = word
        self.patterns: dict[tuple[int, ...], Guess] = {}
        for pattern, matches in optimizer.pattern_matches(word):
            next_solutions = remaining_solutions.intersection(matches)
            if len(next_solutions) > 0 and pattern != (2, 2, 2, 2, 2):
                self.patterns[pattern] = optimizer.optimal_guess(next_solutions)
    
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


class TreeBot(Bot):
    """A Wordle bot that creates an optimal decision tree of guesses for each possible pattern."""
    def __init__(
            self, 
            save_trees: bool=True,
            optimal_tree_file: Path=TREE_FILE,
            word_tree_file_suffix: str='_tree.p'
            ) -> None:
        self.save_trees = save_trees
        self.optimal_tree_file = optimal_tree_file
        self.word_tree_file_suffix = word_tree_file_suffix
        
    def initialize_with_optimizer(
            self,
            optimizer: Optimizer,
            dictionary: list[str],
            possible_solutions: frozenset[str]
            ) -> None:
        TREE_DIRECTORY.mkdir(exist_ok=True)
        self.dictionary = dictionary
        self.possible_solutions = possible_solutions

        if self.save_trees and self.optimal_tree_file.exists():
            print('Loading tree from file')
            self.tree: Guess = pickle.load(self.optimal_tree_file.open('rb'))
            self.height = self.tree.height
            self.width: int = self.tree.width
        else:
            print('Building tree of all possible guesses')
            remaining_solutions = possible_solutions
            if SAVE_TIME:
                self.tree = Guess('scale', remaining_solutions, optimizer)
                self.height = self.tree.height
                self.width: int = self.tree.width
            else:
                self.height = N_GUESSES
                self.width = len(remaining_solutions)
                for word in tqdm(optimizer.sort(possible_solutions, remaining_solutions)):
                    tree_file = TREE_DIRECTORY / (word + self.word_tree_file_suffix)
                    if self.save_trees and tree_file.exists():
                        tree: Guess = pickle.load(tree_file.open('rb'))
                    else:
                        tree = Guess(word, remaining_solutions, optimizer)
                        if self.save_trees:
                            pickle.dump(tree, tree_file.open('wb+'))
                    height = tree.height
                    width: int = tree.width
                    if VERBOSE: print(tree, ":", format_list(tree.words_at_depth(height)), optimizer.status())
                    if height < self.height:
                        self.tree = tree
                        self.height = height
                        self.width = width
                    elif height == self.height and width < self.width:
                        self.tree = tree
                        self.height = height
                        self.width = width
            print('Saving optimal tree to file')
            if self.save_trees:
                pickle.dump(self.tree, self.optimal_tree_file.open('wb+'))
        if VERBOSE: print('Tree starting with', self.tree.word, 'is optimal with height:', self.height, 'and width:', self.width)

    def solve(self, game: Game) -> str | None:
        guess = self.tree
        for n_round in range(N_GUESSES):
            pattern = game.guess(guess.word)

            # Print round information
            if VERBOSE: print('Guessing: ', guess.word, '  Info: ', pattern)
            if pattern == (2, 2, 2, 2, 2):
                if VERBOSE: print(f'WIN IN {n_round + 1} GUESSES!')
                return guess.word

            # Go to the next guess in the tree
            if pattern not in guess.patterns:
                if VERBOSE: print('Failed to find pattern', pattern, 'as possible result after guess:', guess.word)
                return None
            guess: Guess = guess.patterns[pattern]

        return None


class HeightOptimizer(Optimizer):
    """Optimizes the tree for minimizing the height and width of the base."""
    def __init__(
            self, 
            pattern_dict: dict[str, dict[tuple[int, ...], frozenset[str]]]
            ) -> None:
        super().__init__(pattern_dict)
        self.cache: cachetools.LRUCache[tuple[frozenset[str]], Guess] = cachetools.LRUCache(maxsize=MAX_CACHE_SIZE)

    @cachetools.cachedmethod(lambda self: self.cache)
    def optimal_guess(self, remaining_solutions: frozenset[str]) -> Guess:
        best_guess = None
        for word in self.sort(remaining_solutions, remaining_solutions):
            guess = Guess(word, remaining_solutions, self)
            if best_guess is None or guess.height < best_guess.height:
                best_guess = guess
            elif guess.height == best_guess.height and guess.width < best_guess.width:
                best_guess = guess
        return cast(Guess, best_guess)
    
    def status(self) -> str:
        return '(cache={:.1%})'.format(self.cache.currsize / MAX_CACHE_SIZE)


class HTreeBot(TreeBot):
    """A tree optimized for minimizing the height and width of the base."""

    def initialize(
            self,
            dictionary: list[str],
            possible_solutions: frozenset[str],
            pattern_dict: dict[str, dict[tuple[int, ...], frozenset[str]]]
            ) -> None:
        super().initialize(dictionary, possible_solutions, pattern_dict)
        optimizer = HeightOptimizer(pattern_dict)
        super().initialize_with_optimizer(optimizer, dictionary, possible_solutions)


class AverageDepthOptimizer(Optimizer):
    def __init__(
            self, 
            pattern_dict: dict[str, dict[tuple[int, ...], frozenset[str]]]
            ) -> None:
        super().__init__(pattern_dict)
        self.cache: cachetools.LRUCache[tuple[frozenset[str]], Guess] = cachetools.LRUCache(maxsize=MAX_CACHE_SIZE)

    @cachetools.cachedmethod(lambda self: self.cache)
    def optimal_guess(self, remaining_solutions: frozenset[str]) -> Guess:
        best_guess = None
        for word in self.sort(remaining_solutions, remaining_solutions):
            guess = Guess(word, remaining_solutions, self)
            if best_guess is None or guess.total_guesses < best_guess.total_guesses:
                best_guess = guess
            elif guess.total_guesses == best_guess.total_guesses and guess.height < best_guess.height:
                best_guess = guess
            elif guess.total_guesses == best_guess.total_guesses and guess.height == best_guess.height and guess.width < best_guess.width:
                best_guess = guess
        return cast(Guess, best_guess)
    
    def status(self) -> str:
        return '(cache={:.1%})'.format(self.cache.currsize / MAX_CACHE_SIZE)


class ATreeBot(TreeBot):
    """A tree optimized for minimizing the average depth (total number of guesses)."""
    def __init__(
            self, 
            save_trees: bool=True
            ) -> None:
        super().__init__(
            save_trees=save_trees, 
            optimal_tree_file=(TREE_DIRECTORY / 'atree.p'), 
            word_tree_file_suffix='_atree.p')

    def initialize(
            self,
            dictionary: list[str],
            possible_solutions: frozenset[str],
            pattern_dict: dict[str, dict[tuple[int, ...], frozenset[str]]]
            ) -> None:
        super().initialize(dictionary, possible_solutions, pattern_dict)
        optimizer = AverageDepthOptimizer(pattern_dict)
        super().initialize_with_optimizer(optimizer, dictionary, possible_solutions)


if __name__ == "__main__":
    from wordle import GameImpl, generate_pattern_dict
    possible_solutions = frozenset(['crane', 'bears', 'weary'])
    dictionary = list(possible_solutions)
    pattern_dict = generate_pattern_dict(dictionary, possible_solutions)
    bot = ATreeBot(save_trees=False)
    bot.initialize(dictionary, possible_solutions, pattern_dict)
    for word_to_guess in possible_solutions:
        game = GameImpl(word_to_guess, True)
        result = bot.solve(game)
        print('For', word_to_guess, 'took', game.num_guesses(), 'to get:', result)
