"""
Script used to process the Kaggle dataset and extract the matches.
It also creates the vocabulary from it.
https://www.kaggle.com/milesh1/35-million-chess-games
"""


import os
import re
from art import tprint
import questionary


def preprocess_kaggle():
    vocab_counter = set()


    # check if this file has already been preprocessed
    if os.path.exists("./dataset/processed_kaggle.txt"):
        response = questionary.confirm("It appears that the kaggle file has already been preprocessed; reprocess?").ask()
        if not response:
            return
        os.remove("./dataset/processed_kaggle.txt")
    
    with open(f"data/datasets-cleaned/kaggle_cleaned.txt", "w", encoding="utf-8") as outf:
        with open("dataset/kaggle1.txt", "r", encoding="utf-8") as inpf:
            for line in inpf:
                try:
                    ostr = line.split("###")[1].strip()
                    ostr = re.sub("W\d+.", "", ostr)
                    ostr = re.sub("B\d+.", "", ostr)

                    if len(ostr) > 0:
                        if ostr[-1] != '\n':
                            ostr = ostr + '\n'

                        outf.write(ostr)

                        for move in ostr.split(" "):
                            move = move.replace("\n", "")

                            if move != "":
                                vocab_counter.add(move)
                    else:
                        a = 0
                except:
                    pass

        os.makedirs("vocabs", exist_ok=True)

        with open(f"vocabs/kaggle_vocab.txt", "w", encoding="utf-8") as f:
            for v in vocab_counter:
                f.write(v + "\n")
                
                
if __name__ == "__main__":
    tprint("ChessFormers PreProcessor")
    preprocess_kaggle()
