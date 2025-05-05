import os
import argparse
import random
import re
import json
from datetime import datetime

from nltk.grammar import Nonterminal

from collections import defaultdict
from typing import Dict, List, Union, Optional

from tqdm import tqdm

from utils import split_by_punct


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


def reverse_normalization(x: List[str]) -> str:
    """
    This function converts a list of strings (words) into a punctuation- and capitalization-correct string.  
    Example:  
    ['did', 'You', 'close', 'the', 'door', '?'] -> 'Did you close the door?'
    """
    if len(x) == 0:
        return ""
    if len(x) == 1:
        return x[0]
    
    x_splitted = split_by_punct(x)
    txts = [' '.join(s).lower() for s in x_splitted]
    # for s in txts:
    #     s[0] = s[0].upper()
    # re.split(r'(?<=[.!?])', txt)
    # # txt = txt.replace(' .', '.').replace(' ?', '?').replace(' !', '!')
    # # txt = txt.replace(' ,', ',').replace(' :', ':').replace(' ;', ';')
    # # txt = txt.replace(' \'', '\'')
    # txt.split()
    
    return '\n'.join(txts)


def generate(grammar: Grammar_type, do_rnorm: bool = True) -> str:
    """
    Iteratively generate text using a stack-based approach
    """
    stack: List[Union[Nonterminal, str]] = [Nonterminal('TOP')]  # Initialize with start symbol
    parts = []

    while stack:
        current = stack.pop()
        if current in grammar:  # Non-terminal expansion
            expansion = random.choice(grammar[current])
            if isinstance(expansion, list):
                for symbol in reversed(expansion):
                    stack.append(symbol)
            else:
                stack.append(expansion)
        else:  # Terminal symbol
            # parts.append(parser.get_terminal(symbol.name).pattern.value)
            parts.append(current.strip('\"'))
    if do_rnorm:
        return reverse_normalization(parts)
    return ' '.join(parts)


def main(gram_pth: str, output_pth: str, n_iterations: int, seed: Optional[int] = None) -> None:
    pth = os.path.join(gram_pth)
    with open(pth, 'r') as file:
        raw_gram = json.load(file)
    # parser = Lark(open(pth).read(), start='start', parser='lalr')
    print(datetime.now().strftime("%Y-%M-%d %H:%M:%S"), 'Start parsing grammar.')
    grammar = json_to_grammar(raw_gram)
    # parser = load_grammar.load_grammar(open(pth).read(), None, None, False)
    # parser = parser[0]
    # print(parser.term_defs)
    print(datetime.now().strftime("%Y-%M-%d %H:%M:%S"), 'It has been parsed!')
    
    if seed is not None:
        random.seed(seed)
    with open(output_pth, 'a') as file:
        for i in tqdm(range(n_iterations)):
            file.write(generate(grammar))
            if i != n_iterations - 1:
                file.write('\n')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-i', '--input', dest='grammar', type=validate_input_file, required=True, help='Input grammar in JSON format')
    argParser.add_argument('-o', '--output', dest='output', type=validate_output_file, required=True, help='Output text file')
    argParser.add_argument('-n', '--n_iterations', dest='n_iterations', required=True, type=int, help='Number of strings to generate')
    argParser.add_argument('-s', '--seed', dest='seed', required=False, type=int, help='Randsom seed')
    args = argParser.parse_args()
    main(gram_pth=args.grammar, output_pth=args.output, n_iterations=args.n_iterations, seed=args.seed)
