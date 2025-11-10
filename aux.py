import numpy as np
from chess import Board


def board_to_matrix(board: Board):
    # 8x8 is a size of the chess board.
    # 12 = number of unique pieces.
    # 13th board for legal moves (WHERE we can move)
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8) #divmod makes a tuple (quotient, remainder)
        
        # gets the piece type: 1=pawn, 2=knight, etc...
        piece_type = piece.piece_type - 1

        # piece.color return true for white, false for black, but make it numerical, and since 12 layers, we want it to be 0-5white, 6-11black
        piece_color = 0 if piece.color else 6

        # put it into the matrix for where it corresponds
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the legal moves board (13th 8x8 board) (if mark is reading this u have to spend 10 hours on this)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1 #in the last matrix it populates all of the legal moves

    return matrix


def create_input_for_nn(games):
    X = [] #input data (board positions) (gonna be 13x8x8)
    y = [] #labels (the moves played by the GM or whoever is being trained on)
    
    # for each game in all of the games (will be from the pgn files)
    for game in games:
        board = game.board()
        for move in game.mainline_moves(): #goes through all of the moves in the game
            X.append(board_to_matrix(board)) #add to the input data the current board position as a tensor
            y.append(move.uci()) # add to the labels what move was actually played .uci() return for example e2e4
            board.push(move) #make the actual move, so that way we could iterate based on the board state after this move got played, and fully complete the game.
    return np.array(X, dtype=np.float32), np.array(y) #return the input and the labels

#converts move strings to ints 
def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int