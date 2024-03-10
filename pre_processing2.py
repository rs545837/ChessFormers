"""
Script used to process the Kaggle dataset and extract the matches.
It also creates the vocabulary from it.
https://www.kaggle.com/milesh1/35-million-chess-games
"""

import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import re
import questionary
from art import tprint

max_game_length = 512
min_game_length = 20

def preprocess_kingbase():
    print("Now processing kingbase.txt")

    write_folder = "./data/datasets-cleaned/"
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    # check if this file has already been preprocessed
    if os.path.exists("./data/datasets-cleaned/kingbase_cleaned.txt"):
        response = questionary.confirm("It appears that the kingbase file has already been preprocessed; reprocess?").ask()
        if not response:
            return

        os.remove("./data/datasets-cleaned/kingbase_cleaned.txt")

    unprocessed_kingbase_lines = open("./dataset/kingbase1.txt", "r").readlines()

    processed_kingbase_lines = open("./data/datasets-cleaned/kingbase_cleaned.txt", "w")

    line_length = []
    for line in tqdm.tqdm(unprocessed_kingbase_lines):
        split_line = line.split()
        output_line = " ".join(split_line[6:-1]) + "\n"
        output_line = re.sub(r'[0-9]+\.', '', output_line)
        if len(output_line) <= max_game_length and '[' not in output_line and ']' not in output_line:
            processed_kingbase_lines.writelines(output_line)
            line_length.append(len(output_line))

    x = np.array(line_length)



    print("Total games in the post-processed file: %d", len(line_length))               
                
if __name__ == "__main__":
    tprint("ChePT Preprocessor")
    preprocess_kingbase()
