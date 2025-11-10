import os
import time
import torch
import torch.nn as nn
from chess import pgn
from tqdm import tqdm
from aux import create_input_for_nn, encode_moves


def loadPgn(filePath):
    games = []

    with open(filePath, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

files = [file for file in os.listdir("../../data/pgn") if file.endswith(".pgn")]
LIMIT_OF_FILES = min(len(files), 28)

games = []
i = 1
for file in tqdm(files):
    games.extend(loadPgn(f"../../data/pgn/{file}"))
    if i >= LIMIT_OF_FILES:
        break
    i += 1

X, y = create_input_for_nn(games)
X = X[0:2500000]
y = y[0:2500000]