# ChessFormers

This deep transformer architecture observes strings of Portable Game Notation (PGN) - a common string representation of chess games designed for maximum human understanding - and outputs strong predicted moves alongside an English commentary of what the model is trying to achieve.

Dataset Used: [3.5 Million Chess Games dataset](https://www.kaggle.com/datasets/milesh1/35-million-chess-games/data)


## Pre_Processing
The pre_processing.py file outputs the sequence of all the moves taken in the chess games and removes all the unnecessary characters and words from the chess game sequences present in the 3.5-million-chess-games dataset.
This is how the output of the processed_dataset looks like:
<img width="310" alt="Screenshot 2024-03-09 at 12 37 48 PM" src="https://github.com/rs545837/ChessFormers/assets/114828377/a9f513bf-1a64-4a5b-9bd9-5ffb954afc64">
