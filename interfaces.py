from abc import ABC, abstractmethod


class Game(ABC):
    """A game to solve for a single word."""
    @abstractmethod
    def guess(self, word: str) -> tuple[int, ...]:
        """Guess a word, returns the pattern identifying correct letters."""
        pass

    @abstractmethod
    def hard_mode_filter(self, word: str) -> bool:
        """A filter that determines if a guess word is valid."""
        pass


class Bot(ABC):
    """A bot that can play wordle, solving to find words."""
    @abstractmethod
    def initialize(self,
                   dictionary: list[str],
                   possible_solutions: list[str],
                   pattern_dict: dict[str, dict[tuple[int, ...], set[str]]]
                   ) -> None:
        """Initialize the bot, will be called once before any calls to `solve`."""
        pass

    @abstractmethod
    def solve(self, game: Game) -> str | None:
        """Called once per word, return the correct word once solved, or None if not solved."""
        pass
