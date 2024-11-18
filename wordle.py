#!/usr/bin/python
import os
import functools
import itertools
import sqlite3
import signal
from tqdm import tqdm
from collections import defaultdict, Counter

N_GUESSES = 10
DICT_FILE = 'all_words.txt'
SOLUTIONS_FILE = 'words.txt'
DB_FILE = 'pattern_dict.db'
SAVE_TIME = True

def calculate_pattern(guess, true):
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
    wrong = [i for i, v in enumerate(guess) if v != true[i]]
    counts = Counter(true[i] for i in wrong)
    pattern = [2] * 5
    for i in wrong:
        v = guess[i]
        if counts[v] > 0:
            pattern[i] = 1
            counts[v] -= 1
        else:
            pattern[i] = 0
    return tuple(pattern)


def generate_pattern_dict(dictionary, possible_solutions, db_file):
    """For each word and possible information returned, store a list
    of candidate words
    >>> pattern_dict = generate_pattern_dict(['weary', 'bears', 'crane'])
    >>> pattern_dict['crane'][(2, 2, 2, 2, 2)]
    {'crane'}
    >>> sorted(pattern_dict['crane'][(0, 1, 2, 0, 1)])
    ['bears', 'weary']
    """
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.execute('CREATE TABLE patterns (word TEXT NOT NULL, pattern TEXT NOT NULL, matches TEXT NOT NULL);')
    for word in tqdm(dictionary):
        for solution in possible_solutions:
            pattern = calculate_pattern(word, solution)
            cur.execute('INSERT INTO patterns VALUES(?, ?, ?)',
                        (word, ''.join(map(str, pattern)), solution))
        con.commit()
    cur.execute('CREATE INDEX pattern_index ON patterns(word, pattern);')
    cur.execute('CREATE INDEX matches_index ON patterns(matches);')
    con.close()


def get_pattern_dict_macthes(pattern_dict, word, pattern):
    cur = pattern_dict.cursor()
    matches = set()
    for row in cur.execute('SELECT matches FROM patterns WHERE word=? AND pattern=?',
                           (word, ''.join(map(str, pattern)))):
        matches.add(row[0])
    return matches

def calculate_entropies(words, remaining_solutions, pattern_dict):
    """Calculate the entropy for every word in `words`, taking into account
    the `remaining_solutions`"""
    print('Calculating entropies for ', len(words), ' words')

    cur = pattern_dict.cursor()
    cur.execute('DROP TABLE IF EXISTS possible_words;')
    cur.execute('CREATE TABLE possible_words (word TEXT NOT NULL PRIMARY KEY);')
    cur.executemany('INSERT INTO possible_words VALUES(?)', zip(remaining_solutions))
    pattern_dict.commit()

    cur.execute('DROP TABLE IF EXISTS matches;')
    cur.execute("""
        CREATE TABLE matches AS
          SELECT
            p.word AS word,
            COUNT(*) AS num_matches
          FROM patterns AS p
            INNER JOIN possible_words AS w
              ON p.matches = w.word
          GROUP BY p.word, pattern;""")
    cur.execute('CREATE INDEX matches_words ON matches(word);')
    pattern_dict.commit()

    return [(row[0], row[1]) for row in cur.execute("""
        SELECT
          raw.word,
          -SUM((num_matches/total_matches) * LOG(num_matches/total_matches)) AS entropy
        FROM 
          matches AS raw
          INNER JOIN
            (SELECT
              word,
              CAST(SUM(num_matches) AS REAL) AS total_matches
            FROM matches
            GROUP BY word
            ) AS totals
            ON raw.word = totals.word
        GROUP BY raw.word
        ORDER BY entropy DESC""")]

abort = False

def main():
    # load all 5-letter-words for making patterns 
    with open(DICT_FILE) as ifp:
        all_dictionary = list(map(lambda x: x.strip(), ifp.readlines()))

    # Load 2315 words for solutions
    with open(SOLUTIONS_FILE) as ifp:
        possible_solutions = list(map(lambda x: x.strip(), ifp.readlines()))

    # Calculate the pattern_dict database if needed.
    if DB_FILE not in os.listdir('.'):
        generate_pattern_dict(all_dictionary, possible_solutions, DB_FILE)

    pattern_dict = sqlite3.connect(DB_FILE)

    # Cache of first round info's and the resulting guess.
    initial_cache = {}

    # Overall accumulated stats across all words.
    stats = defaultdict(list)

    # Don't quit immediately on Ctrl-C, finish the iteratation and print the results.
    def stop(signum, frame):
        print('Aborting after this iteration')
        global abort
        abort = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    for word_to_guess in tqdm(possible_solutions):
        if abort:
            break

        if SAVE_TIME:
            guess_word = 'soare'
            remaining_solutions = set(possible_solutions)
            info = calculate_pattern(guess_word, word_to_guess)
            print('Guessing: ', guess_word, '  Info: ', info)
            matches = get_pattern_dict_macthes(pattern_dict, guess_word, info)
            remaining_solutions = remaining_solutions.intersection(matches)
            init_round = 1
        else:
            remaining_solutions = set(possible_solutions)
            init_round = 0

        for n_round in range(init_round, N_GUESSES):
            print('Round: ', n_round + 1, '  Remaining words: ', len(remaining_solutions))
            if len(remaining_solutions) > 2:
                if n_round == 1 and info in initial_cache:
                  print('Using previously found guess')
                  guess_word = initial_cache[info]
                else:
                  candidates = all_dictionary
                  entropies = calculate_entropies(candidates, remaining_solutions, pattern_dict)

                  if entropies[0][1] < 0.1:
                      candidates = remaining_solutions
                      entropies = calculate_entropies(candidates, remaining_solutions, pattern_dict)

                  # Guess the candiate with highest entropy, preferring possible solutions to break ties
                  def sorting(first, second):
                      if first[1] < second[1]:
                          return -1
                      if first[1] > second[1]:
                          return 1
                      if first[0] in remaining_solutions and second[0] not in remaining_solutions:
                          return 1
                      if second[0] in remaining_solutions and first[0] not in remaining_solutions:
                          return -1
                      return 0
                  entropies.sort(key=functools.cmp_to_key(sorting), reverse=True)
                  print('Top guesses: ', [(word, '{:.5f}'.format(entropy))
                                          for word, entropy in entropies[:6]])
                  guess_word = entropies[0][0]
                  if n_round == 1:
                      initial_cache[info] = guess_word
            else:
                # One or two words remaining, choose the one, or pick randomly (can't do any better).
                guess_word = list(remaining_solutions)[0]
            info = calculate_pattern(guess_word, word_to_guess)

            # Print round information
            print('Guessing: ', guess_word, '  Info: ', info)
            if guess_word == word_to_guess:
                print(f'WIN IN {n_round + 1} GUESSES!\n\n\n')
                stats['guesses'].append(n_round + 1)
                if n_round + 1 > 6:
                    stats['misses'].append(word_to_guess)
                break

            # Filter our list of remaining possible words
            matches = get_pattern_dict_macthes(pattern_dict, guess_word, info)
            remaining_solutions = remaining_solutions.intersection(matches)
        else:
            stats['misses'].append(word_to_guess)

    print(len(stats['guesses']), 'successful guesses')
    print('Average guess count:', 1.0 * sum(stats['guesses']) / len(stats['guesses']))
    print('Guess counts of')
    print('  1:', len([x for x in stats['guesses'] if x == 1]))
    print('  2:', len([x for x in stats['guesses'] if x == 2]))
    print('  3:', len([x for x in stats['guesses'] if x == 3]))
    print('  4:', len([x for x in stats['guesses'] if x == 4])) 
    print('  5:', len([x for x in stats['guesses'] if x == 5])) 
    print('  6:', len([x for x in stats['guesses'] if x == 6]))
    print(' 7+:', len([x for x in stats['guesses'] if x > 6]))
    print('Missed words:', stats['misses'])

if __name__ == "__main__":
    main()
