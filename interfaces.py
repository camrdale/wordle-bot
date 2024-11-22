from abc import ABC, abstractmethod
from typing import Any


class Game(ABC):
    """A game to solve for a single word."""
    @abstractmethod
    def guess(self, word: str, cache_hit: bool=False) -> tuple[int, ...]:
        """Guess a word, returns the pattern identifying correct letters."""
        pass

    @abstractmethod
    def hard_mode_filter(self, word: str) -> bool:
        """A filter that determines if a guess word is valid."""
        pass


class CancellationWatcher:
    """A mutable object that can be passed by reference to signal cancellation of the call."""

    def __init__(self) -> None:
        self.is_cancelled = False

    def cancel(self) -> None:
        self.is_cancelled = True

    def stop(self, signum: int, frame: Any) -> None:
        self.cancel()


class Bot(ABC):
    """A bot that can play wordle, solving to find words."""
    def __init__(self, cancellation_watcher: CancellationWatcher) -> None:
        self.cancellation_watcher = cancellation_watcher

    @abstractmethod
    def initialize(self,
                   dictionary: list[str],
                   possible_solutions: frozenset[str],
                   pattern_dict: dict[str, dict[tuple[int, ...], frozenset[str]]]
                   ) -> None:
        """Initialize the bot, will be called once before any calls to `solve`."""
        pass

    @abstractmethod
    def solve(self, game: Game) -> str | None:
        """Called once per word, return the correct word once solved, or None if not solved."""
        pass
