# Wordle-Bots

A collection of bots to play Wordle, using different strategies.

## Differences

Some differences from other Wordle bots are that the bots are configured to use
hard mode, and are aware of the shorter list of all possible solution words.

### Hard Mode

Hard mode is an option when playing Wordle (off by default). It forces
subsequent guesses to take into account the information learned from previous
guesses. For example, if the result of guessing STORE was that the S is green
(correct and in the correct place), then all subsequent guesses must start with
S. If the result of guessing STORE is that the S is yellow (present in the
solution but in the wrong place), then all subsequent guesses must include an S
somewhere (can still start with S). If the result of guessing SHINS is the first
S is green and the last S is yellow, then all subsequent guesses must include
two S: one in the first letter, and another one somewhere else.

Using hard mode has the effect of making the bot less efficient, as it
eliminates the possibility of using guesses just to gain information. So the
average number of guesses required to complete all possible solutions is higher
in hard mode. But it also makes the problem more tractable, as the number of
possibilities that can be guessed on subsequent guesses is much smaller.

One interesting note is that the best starting word found by using some methods
(like entropy-based ones), SOARE, is not a good guess to guarantee the finding
of all words in 6 or less guesses when Hard Mode is enabled. This guess can
result in a pattern of green for the S, R and E, and yellow for the O. The
remaining words after this guess and pattern are: STORE, SWORE, SCORE, SPORE,
SHORE, SNORE. So there's no way to guearantee finding the solution, as each
subeqeuent guess can only eliminate one word from the possible solutions.

### Possible Solutions

The original Wordle contained a list of 2,315 possible solutions, in addition to
the list of 13k words that are legal guesses. Some bots don't make use of this
list of possible solutions, as it is not necessarily part of the game. But it is
used by the official Wordle bot.

## Bots

### Entropy Bot (`entropy`)

An entropy-based strategy to wordle, originally inspired by
[3Blue1Brown](https://www.youtube.com/watch?v=v68zYyaEmEA), and forked from
<https://github.com/GillesVandewiele/Wordle-Bot>.

In addition to changing the original bot to support hard mode and to use the
list of possible solutions, some other changes were made. Caching was added to
speed up the lookups, so that if the same sequence of patterns occurred, the
entropy calculations didn't need to be redone. This replaced the original
hard-coding of the best first guess to use, as now it is calculated once for the
first word it is asked to solve for, and then reused on subsequent solves. The
bot was also made deterministic by breaking ties in entropy deterministically.

### Random Bot (`rando`)

A terrible Wordle bot that chooses words randomly. With hard mode disabled, this
bot is particularly bad. But with hard mode enabled, it gets a surprising number
of the solutions correct in 6 guesses (about two thirds).

### Largest Remaining Group Bot (`g-small`)

A Wordle bot that minimizes the largest remaining group after a guess.

### More Groups Bot (`g-more`)

A Wordle bot that maximizes the number of remaining groups after a guess.

### Height Tree Bot (`h-tree`)

A Wordle bot that creates an optimal decision tree of guesses for each possible
resulting pattern.

Each node in the tree represents a guess. The guess has a list of the possible
patterns (edges) that can result from this guess, which each have an optimal
next guess associated with them, and are the nodes directly below the node in
the tree.

The tree is optimized to minimize the height (the number of edges/patterns
present in the longest path connecting the root node to a leaf node) and width
of the base of the tree (the number of nodes that share the longest path
length). This results in a bot that gets the most possible words correct in the
smallest number of guesses. Some of the other bots fail to get some words,
requiring 7 guesses on a few of them.

In addition to the Hard Mode restrictions, the tree bot also limits guesses to
possible solutions, instead of using all possible legal guesses in the Wordle
dictionary. This was done to speed up the process of optimizing trees, as
searching through 13k words to find the optimal one was too time consuming.

### Average Depth Tree Bot (`a-tree`)

Similar to the Height Tree Bot, but instead optimizes the tree to minimize the
total number of guesses it takes to guess every word in the tree. This is the
same as minimizing the average depth of the tree, or minimizing the average
number of guesses across all possible solutions.

### Ideal Tree Bot (`i-tree`)

Another tree bot combining the Height Tree Bot and Average Depth Tree Bot. It
optimizes the tree primarily not to fail any words (use more than 6 guesses).
Within that constraint, it optimizes to minimize the total number of guesses it
takes to guess every word in the tree.

### Dictionary Tree Bot (`i-tree`)

Similar to the Ideal Tree Bot, but this tree bot expands the possible guesses to
all legal guesses in the Wordle dictionary, not just the possible solutions. The
optimization of the tree is still primarily not to fail any words, and then to
minimize the total number of guesses it takes to guess every word in the
possible solutions. Initially, this took 24 hours to process 1% of the tree for
a single starting word. To speed it up, instead it only considers the 100
possible legal guesses that have the highest entropy at each iteration in the
tree.

## Install & Run

1. Clone the script:

```bash
git clone https://github.com/camrdale/wordle-bot.git
```

2. Install the needed dependencies

```bash
sudo apt install python3-scipy python3-tqdm python3-cachetools
```

3. Run the script (it will take a long time to run all the bots)

```bash
python3 wordle.py
```

## Statistics

Letting all the bots run over all 2,315 possible solutions, results in this:

|                 **Bot** | `d-tree` | `i-tree` | `a-tree` | `h-tree` | `g-more` | `g-small` | `entropy` | `rando` |
| ----------------------: | -------: | -------: | -------: | -------: | -------: | --------: | --------: | ------: |
|     **Initialize time** |     ~50h |     ~48h |     ~60h |     ~60h |   1m 39s |    1m 39s |    1m 39s |      0s |
|          **Solve time** |       0s |       0s |       0s |       0s |   1m 40s |    1m 42s |    1m 41s |     38s |
| **Average guess count** |    3.509 |    3.554 |    3.518 |    3.642 |    3.527 |     3.672 |     3.603 |   6.443 |
|  **Words guessed in 1** |        0 |        1 |        1 |        1 |        1 |         1 |         0 |       0 |
|                   **2** |      122 |      130 |      146 |      110 |      122 |       104 |        77 |      35 |
|                   **3** |     1057 |      991 |     1015 |      829 |     1085 |       891 |      1043 |      89 |
|                   **4** |      986 |      998 |      986 |     1153 |      912 |      1040 |       972 |     223 |
|                   **5** |      135 |      169 |      143 |      221 |      164 |       228 |       178 |     270 |
|                   **6** |       15 |       26 |       20 |        1 |       22 |        40 |        38 |     309 |
|                  **7+** |        0 |        0 |        4 |        0 |        9 |        11 |         7 |     843 |
|       **Starting word** |    SALET |    PLATE |    SLATE |    CLASP |    TRACE |     RAISE |     SOARE |     N/A |

Note that only the Dictionary Tree Bot, Ideal Tree Bot and Height Tree Bot were
able to succeed with all words in 6 guesses, which is because they are optimized
for that use case. Here are the words that were missed by the other bots (first
10 only):

- `a-tree`: baste, daunt, boxer, batch
- `g-more`: dilly, goner, foyer, saner, folly, boxer, batch, gaunt, catch
- `g-small`: latch, grape, lower, caste, homer, shake, hover, shade, shore,
  crown
- `entropy`: daunt, hatch, golly, shore, found, batch, gaunt
- `rando`: oaken, decal, gloss, briny, chart, elbow, merry, bezel, mouth, humid
