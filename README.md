# ChessFormers

This deep transformer architecture observes strings of Portable Game Notation (PGN) - a common string representation of chess games designed for maximum human understanding - and outputs strong predicted moves alongside an English commentary of what the model is trying to achieve.

Dataset Used: [3.5 Million Chess Games dataset](https://www.kaggle.com/datasets/milesh1/35-million-chess-games/data) and the [KingBase Chess Dataset](https://archive.org/details/KingBase2019)


## Pre_Processing
The pre_processing.py file outputs the sequence of all the moves taken in the chess games and removes all the unnecessary characters and words from the chess game sequences present in the 3.5-million-chess-games dataset.

***This is how the output of the pre_processing on 3.5 Million Chess Games looks like:***
<img width="620" alt="Screenshot 2024-03-09 at 12 37 48 PM" src="https://github.com/rs545837/ChessFormers/assets/114828377/a9f513bf-1a64-4a5b-9bd9-5ffb954afc64">
> [!NOTE]  
> Each game occupying a single line.

***This is how the output of the pre_processing on KingBase Games looks like:***

<img width="318" alt="Screenshot 2024-03-11 at 5 38 16 PM" src="https://github.com/rs545837/ChessFormers/assets/114828377/a3310ba9-997a-42e9-8138-54380102e263">

> [!NOTE]  
> A single game in multiple lines.

## Causal SelfAttention Module 
Causal self-attention ensures that the outputs for a certain position in a sequence is based only on the known outputs at previous positions and not on future positions. In simpler terms, it ensures that the prediction for each next word should only depend on the preceding words. To achieve this in GPT-like LLMs, for each token processed, we mask out the future tokens, which come after the current token in the input text.

<img width="492" alt="Screenshot 2024-03-13 at 11 05 33 AM" src="https://github.com/rs545837/ChessFormers/assets/114828377/a7a89776-0191-4f82-a931-477afe756425">

## Model Architecture
## FineTuning Module
### Finetune_Early, Finetune_Middle, Finetune_Late, Commentary_Dataset, PretrainDataset:
These classes inherit from the torch.utils.data.Dataset class, making them compatible with PyTorch's DataLoader.
Each class represents a different dataset for specific stages of finetuning or pretraining.
The `__init__` method initializes the dataset with relevant parameters and processes the input data.
The `__len__` method returns the length of the dataset.
The `__getitem__` method retrieves an item from the dataset at a given index.

### Directory:
This class acts as a directory or factory for creating different datasets based on the version specified.
The `__init__` method stores information about the dataset, version, configuration arguments, and pretraining vocabulary.
The `__call__` method dynamically selects the appropriate dataset class based on the specified version and returns an instance of that class.

### PretrainDataset:
This class represents the dataset used for the initial pretraining stage.
It prepares the data by encoding it into tensors of integers based on a character-to-index mapping.
The `__getitem__` method returns input-output pairs, where each input and output are subsequences of the original text shifted by one position.

### finetune_versions:
This dictionary maps version numbers to the corresponding finetuning dataset classes.

### Main Section:
In the main section, a sample of game data is loaded from a file.
An instance of the PretrainDataset class is created with the sample data.
The `__main__` block demonstrates how to use the code, creating and printing an instance of the PretrainDataset class.

## Trainer Module
## Top Module (To Run the whole Project)
## Analysis Module 
