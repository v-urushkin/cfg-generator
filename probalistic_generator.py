import os
import argparse
import random
import re
import csv
import json
from datetime import datetime
from pathlib import Path

from nltk.grammar import Nonterminal

from collections import defaultdict
from typing import Dict, List, Union, Optional, Set, Tuple

from tqdm import tqdm

from utils import preprocess


Grammar_type = Dict[Nonterminal, List[Union[List[Nonterminal], str]]]


def validate_input_file(grammar):
    if not os.path.exists(grammar):
        raise argparse.ArgumentTypeError("{0} does not exist".format(grammar))
    return grammar


def validate_output_file(output):
    if os.path.exists(output):
        raise argparse.ArgumentTypeError("{0} does already exist".format(output))
    return output


def json_to_grammar(raw_gram: Dict[str, List[str]]) -> Grammar_type:
    raw_left_parts = list(raw_gram.keys())
    grammar_as_dict = defaultdict(list)
    for raw_lhs in tqdm(raw_left_parts):
        lhs = Nonterminal(raw_lhs)
        for rhs in raw_gram[raw_lhs]:
            rhs_list_raw = rhs.split(' ')
            if len(rhs_list_raw) == 1:
                symb = rhs_list_raw[0]
                if symb.startswith("\""):
                    grammar_as_dict[lhs].append(symb)
                else:
                    grammar_as_dict[lhs].append([Nonterminal(symb)])
            else:
                rhs_nt_list = []
                for symb in rhs_list_raw:
                    rhs_nt_list.append(Nonterminal(symb))
                grammar_as_dict[lhs].append(rhs_nt_list)
    return grammar_as_dict


def reverse_normalization(x: List[str], NNP: Set[Optional[str]] = set()) -> str:
    """
    This function converts a list of strings (words) into a punctuation- and capitalization-correct string.  
    Example:  
    ['did', 'You', 'close', 'the', 'door', '?'] -> 'Did you close the door?'
    """
    if len(x) == 0:
        return ""
    if len(x) == 1:
        return x[0]
    
    x_splitted = preprocess(x, NNP)
    txts = [' '.join(s) for s in x_splitted]
    txt = ' '.join(txts)
    txt = txt.replace(' `` ', ' ``').replace(' ` ', '` ')
    txt = txt.replace(' .', '.').replace(' ?', '?').replace(' !', '!')
    txt = txt.replace(' ,', ',').replace(' :', ':').replace(' ;', ';')
    txt = txt.replace(' \'', '\'').replace(' - ', '-')
    txt = txt.replace('_lrb_ ', '(').replace(' _rrb_', ')')
    txt = txt.replace('_lcb_ ', '(').replace(' _rcb_', ')')
    txt = txt.replace('_lsb_ ', '(').replace(' _rsb_', ')')
    
    return txt


def generate(grammar: Grammar_type, do_rnorm: bool = True, NNP: Set[Optional[str]] = set()) -> Tuple[str, int]:
    """
    Iteratively generate text using a stack-based approach
    """
    stack: List[Union[Nonterminal, str]] = [Nonterminal('TOP')]  # Initialize with start symbol
    parts = []
    complexity = 0
    while stack:
        current = stack.pop()
        if current in grammar:  # Non-terminal expansion
            expansion = random.choice(grammar[current])
            if isinstance(expansion, list):
                for symbol in reversed(expansion):
                    stack.append(symbol)
            else:
                stack.append(expansion)
            complexity += 1
        else:  # Terminal symbol
            # parts.append(parser.get_terminal(symbol.name).pattern.value)
            parts.append(current.strip('\"'))
    if do_rnorm:
        return reverse_normalization(parts, NNP), complexity
    return ' '.join(parts), complexity


def main(gram_pth: str, output_pth: str, n_iterations: int, seed: Optional[int] = None) -> None:
    pth = os.path.join(gram_pth)
    with open(pth, 'r') as file:
        raw_gram = json.load(file)
    # parser = Lark(open(pth).read(), start='start', parser='lalr')
    print(datetime.now().strftime("%Y-%M-%d %H:%M:%S"), 'Start parsing grammar.')
    grammar = json_to_grammar(raw_gram)
    nnp_set = set([p.strip('\"') for p in grammar[Nonterminal('NNP')]])
    nnps_set = set([p.strip('\"') for p in grammar[Nonterminal('NNPS')]])
    nnp_nnps_set = nnp_set | nnps_set
    
    print(f"NNP len = {len(nnp_set)}")
    print(f"NNPS len = {len(nnps_set)}")
    print(datetime.now().strftime("%Y-%M-%d %H:%M:%S"), 'It has been parsed!')
    
    if seed is not None:
        random.seed(seed)
    
    output_path = Path(output_pth)
    file_counter = 1
    new_filename = output_path.parent / f"{output_path.stem}_{file_counter - 1}{output_path.suffix}"
    current_row_count = 0
    rows_per_file = 1_000_000  # Maximum data rows per file

    # Open the initial file
    current_file = open(new_filename, mode='w', newline='', encoding='utf-8')
    writer = csv.writer(current_file, quoting=csv.QUOTE_STRINGS)
    writer.writerow(['text', 'complexity'])

    try:
        for i in tqdm(range(n_iterations)):
            generation = generate(grammar, NNP=nnp_nnps_set)
            writer.writerow(generation)
            current_row_count += 1

            if current_row_count >= rows_per_file and i != (n_iterations - 1):
                # Close current file and open a new one
                current_file.close()
                file_counter += 1
                new_filename = output_path.parent / f"{output_path.stem}_{file_counter - 1}{output_path.suffix}"
                current_file = open(new_filename, mode='w', newline='', encoding='utf-8')
                writer = csv.writer(current_file, quoting=csv.QUOTE_STRINGS)
                writer.writerow(['text', 'complexity'])
                current_row_count = 0  # Reset counter for the new file
    finally:
        current_file.close()  # Ensure the last file is closed

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-i', '--input', dest='grammar', type=validate_input_file, required=True, help='Input grammar in JSON format')
    argParser.add_argument('-o', '--output', dest='output', type=validate_output_file, required=True, help='Output text file')
    argParser.add_argument('-n', '--n_iterations', dest='n_iterations', required=True, type=int, help='Number of strings to generate')
    argParser.add_argument('-s', '--seed', dest='seed', required=False, type=int, help='Randsom seed')
    args = argParser.parse_args()
    main(gram_pth=args.grammar, output_pth=args.output, n_iterations=args.n_iterations, seed=args.seed)
