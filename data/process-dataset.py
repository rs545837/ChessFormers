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
    print("Now processing kingbase-ftfy.txt")

    write_folder = "./data/datasets-cleaned/"
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    # check if this file has already been preprocessed
    if os.path.exists("./data/datasets-cleaned/kingbase_cleaned.txt"):
        response = questionary.confirm("It appears that the kingbase file has already been preprocessed; reprocess?").ask()
        if not response:
            return

        os.remove("./data/datasets-cleaned/kingbase_cleaned.txt")

    unprocessed_kingbase_lines = open("./dataset/kingbase-ftfy.txt", "r", errors="replace").readlines()

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

    plt.hist(x, density=True, bins=100)  # density=False would make counts
    plt.ylabel('Relative Frequency')
    plt.xlabel('Sequence Length')
    plt.show()

    print("Total games in the post-processed file: %d", len(line_length))


def preprocess_kaggle():
    print("Now preprocessing kaggle.txt")

    write_folder = "./data/datasets-cleaned/"
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    # check if this file has already been preprocessed
    if os.path.exists("./data/datasets-cleaned/kaggle_cleaned.txt"):
        response = questionary.confirm("It appears that the kaggle file has already been preprocessed; reprocess?").ask()
        if not response:
            return
        os.remove("./data/datasets-cleaned/kaggle_cleaned.txt")

    unprocessed_kaggle_lines = open("./dataset/kaggle.txt", "r", errors="replace").readlines()[5:]

    processed_kaggle_lines = open("./data/datasets-cleaned/kaggle_cleaned.txt", "w")

    line_length = []
    for line in tqdm.tqdm(unprocessed_kaggle_lines):
        split_line = line.split()
        for index, token in enumerate(split_line):
            if index % 2 == 0:
                split_line[index] = token[3:]
            else:
                split_line[index] = token[1:]
        output_line = " ".join(split_line[17:]) + "\n"
        if output_line == "\n":
            continue
        output_line = re.sub(r'[0-9]*\.', '', output_line)
        if len(output_line) <= max_game_length and len(output_line) >= min_game_length and '[' not in output_line and ']' not in output_line:
            processed_kaggle_lines.writelines(output_line)
            line_length.append(len(output_line))

    x = np.array(line_length)

    plt.hist(x, density=True, bins=100)  # density=False would make counts
    plt.ylabel('Relative Frequency')
    plt.xlabel('Sequence Length')
    plt.show()


def main():
    tprint("ChePT   Preprocessor")
    preprocess_kingbase()
    preprocess_kaggle()


if __name__ == '__main__':
    main()