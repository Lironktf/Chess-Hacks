"""Direct engine performance test - no interface overhead"""
import torch
import torch.nn as nn
import chess
import numpy as np
import time

# NNUE Model
class NNUE(nn.Module):
    def __init__(self, input_size=640, hidden1_size=1408, hidden2_size=32):
        super(NNUE, self).__init__()
        self.ft_white = nn.Linear(input_size, hidden1_size)
        self.ft_black = nn.Linear(input_size, hidden1_size)
        self.fc1 = nn.Linear(hidden1_size * 2, hidden2_size)
        self.fc2 = nn.Linear(hidden2_size, 1)
        self.clipped_relu = lambda x: torch.clamp(torch.relu(x), 0, 1)

    def forward(self, white_features, black_features):
        w = self.clipped_relu(self.ft_white(white_features))
        b = self.clipped_relu(self.ft_black(black_features))
        x = torch.cat([w, b], dim=1)
        x = self.clipped_relu(self.fc1(x))
        return self.fc2(x)

# HalfKP feature extraction
def board_to_halfkp_features(board):
    HALFKP_INPUT_SIZE = 640
    white_features = np.zeros(HALFKP_INPUT_SIZE, dtype=np.float32)
    black_features = np.zeros(HALFKP_INPUT_SIZE, dtype=np.float32)

    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)

    if white_king_sq is None or black_king_sq is None:
        return white_features, black_features

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.piece_type == chess.KING:
            continue

        piece_idx = piece.piece_type - 1
        color_offset = 0 if piece.color == chess.WHITE else 5
        idx = (piece_idx + color_offset) * 64 + square
        white_features[idx] = 1.0

        mirrored_square = chess.square_mirror(square)
        mirrored_color_offset = 0 if piece.color == chess.BLACK else 5
        idx_black = (piece_idx + mirrored_color_offset) * 64 + mirrored_square
        black_features[idx_black] = 1.0

    return white_features, black_features

# Piece values and tables
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_TABLE_MIDDLEGAME = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

def evaluate_position_fast(board):
    """Fast evaluation - material + PST only (no NNUE)"""
    material_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES[piece.piece_type]
            sq = square if piece.color == chess.WHITE else chess.square_mirror(square)

            if piece.piece_type == chess.PAWN:
                value += PAWN_TABLE[sq]
            elif piece.piece_type == chess.KNIGHT:
                value += KNIGHT_TABLE[sq]
            elif piece.piece_type == chess.BISHOP:
                value += BISHOP_TABLE[sq]
            elif piece.piece_type == chess.KING:
                value += KING_TABLE_MIDDLEGAME[sq]

            if piece.color == chess.WHITE:
                material_score += value
            else:
                material_score -= value

    if board.is_check():
        material_score += 50 if board.turn else -50

    return material_score

def evaluate_position(model, board, device, use_nnue=True):
    """Hybrid evaluation: material + PST + NNUE"""
    if board.is_checkmate():
        return -20000 if board.turn else 20000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # Fast material evaluation
    material_score = evaluate_position_fast(board)

    # Only use NNUE at leaf nodes
    if not use_nnue:
        return material_score

    # NNUE adjustment
    white_feat, black_feat = board_to_halfkp_features(board)
    white_tensor = torch.tensor(white_feat, dtype=torch.float32).unsqueeze(0).to(device)
    black_tensor = torch.tensor(black_feat, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        nnue_score = model(white_tensor, black_tensor).item()
    nnue_adjustment = nnue_score * 30

    total_score = material_score + nnue_adjustment
    return total_score if board.turn == chess.WHITE else -total_score

def quiescence_search(model, board, alpha, beta, device, qdepth=0):
    """Search all captures to avoid horizon effect"""
    # Only use NNUE at first q-node (depth 0)
    use_nnue = (qdepth == 0)
    stand_pat = evaluate_position(model, board, device, use_nnue=use_nnue)

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -quiescence_search(model, board, -beta, -alpha, device, qdepth + 1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

    return alpha

def alpha_beta(model, board, depth, alpha, beta, device, stats):
    """Alpha-beta pruning with quiescence search"""
    stats['nodes'] += 1

    if depth == 0:
        return quiescence_search(model, board, alpha, beta, device, 0)

    if board.is_game_over():
        return evaluate_position(model, board, device, use_nnue=True)

    best_score = float('-inf')

    for move in board.legal_moves:
        board.push(move)
        score = -alpha_beta(model, board, depth - 1, -beta, -alpha, device, stats)
        board.pop()

        best_score = max(best_score, score)
        alpha = max(alpha, score)

        if alpha >= beta:
            break

    return best_score

def get_best_move(model, board, depth, device):
    """Find best move using iterative deepening"""
    best_move = None
    best_score = float('-inf')

    for d in range(1, depth + 1):
        current_best_move = None
        current_best_score = float('-inf')
        stats = {'nodes': 0}

        for move in board.legal_moves:
            board.push(move)
            score = -alpha_beta(model, board, d - 1, float('-inf'), float('inf'), device, stats)
            board.pop()

            if score > current_best_score:
                current_best_score = score
                current_best_move = move

        if current_best_move:
            best_move = current_best_move
            best_score = current_best_score

        print(f"Depth {d}: move={best_move.uci() if best_move else 'none'}, score={best_score:.0f}, nodes={stats['nodes']:,}")

    return best_move, best_score

# Main test
print("=" * 70)
print("DIRECT ENGINE TEST - No Interface Overhead")
print("=" * 70)
print()

# Load model
device = torch.device("cpu")
model = NNUE()
model_path = "../compact/nnue_best.pth"
state_dict = torch.load(model_path, map_location=device)

# Handle naming convention
if "ft_w.weight" in state_dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == "ft_w.weight":
            new_state_dict["ft_white.weight"] = value
        elif key == "ft_w.bias":
            new_state_dict["ft_white.bias"] = value
        elif key == "ft_b.weight":
            new_state_dict["ft_black.weight"] = value
        elif key == "ft_b.bias":
            new_state_dict["ft_black.bias"] = value
        else:
            new_state_dict[key] = value
    state_dict = new_state_dict

model.load_state_dict(state_dict)
model.eval()
model.to(device)

print(f"✓ NNUE model loaded from {model_path}\n")

# Test positions
test_positions = [
    ("Starting position", chess.Board()),
    ("After 1.e4 e5 2.Nf3 Nc6", chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")),
    ("Ruy Lopez", chess.Board("r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4")),
    ("Sicilian", chess.Board("rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 5")),
]

for desc, board in test_positions:
    print(f"Testing: {desc}")
    print(f"FEN: {board.fen()}")
    print()

    start = time.time()
    best_move, score = get_best_move(model, board, depth=5, device=device)
    elapsed = time.time() - start

    print()
    print(f"  Best move: {best_move.uci()}")
    print(f"  Score: {score:.0f}")
    print(f"  Time: {elapsed:.2f} seconds")

    if elapsed > 30:
        print("  ⚠️  TOO SLOW!")
    elif elapsed > 15:
        print("  ⚠️  Slow")
    elif elapsed > 5:
        print("  ⚠️  Acceptable")
    else:
        print("  ✓ Good")
    print()
    print("-" * 70)
    print()

print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
