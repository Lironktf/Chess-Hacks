from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from chess import Board, Move
from model import Model
from aux import board_to_matrix
import os
import json

app = Flask(__name__)

# Global variables for model and vocabulary
model = None
move_to_int = None
int_to_move = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_vocab():
    """Load the trained model and vocabulary"""
    global model, move_to_int, int_to_move
    
    # Try to find a saved model
    model_path = None
    vocab_path = None
    
    # Check for common model file names
    possible_model_paths = [
        "TORCH_100EPOCHS.pth",
        "checkpoints/checkpoint_stage4_80k_epoch_50.pth",
        "checkpoints/checkpoint_stage3_32k_epoch_30.pth",
        "checkpoints/checkpoint_stage2_16k_epoch_20.pth",
        "checkpoints/checkpoint_stage1_8k_epoch_10.pth",
    ]
    
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        # Look for any .pth file in checkpoints
        if os.path.exists("checkpoints"):
            checkpoints = [f for f in os.listdir("checkpoints") if f.endswith(".pth")]
            if checkpoints:
                checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join("checkpoints", x)), reverse=True)
                model_path = os.path.join("checkpoints", checkpoints[0])
    
    if model_path is None:
        raise FileNotFoundError("No trained model found. Please train a model first using train.py")
    
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get vocabulary
    if 'move_to_int' in checkpoint:
        move_to_int = checkpoint['move_to_int']
    else:
        # Try to load from JSON file
        vocab_json = model_path.replace('.pth', '.json').replace('checkpoint_', 'move_to_int_')
        if os.path.exists(vocab_json):
            with open(vocab_json, 'r') as f:
                move_to_int = json.load(f)
        else:
            raise FileNotFoundError("Vocabulary not found. Please ensure vocabulary is saved with the model.")
    
    # Create reverse mapping
    int_to_move = {idx: move for move, idx in move_to_int.items()}
    
    # Get number of classes
    if 'num_classes' in checkpoint:
        num_classes = checkpoint['num_classes']
    else:
        num_classes = len(move_to_int)
    
    # Initialize and load model
    model = Model(numClasses=num_classes).to(device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()  # Set to evaluation mode
    print(f"Model loaded successfully with {num_classes} move classes")
    print(f"Using device: {device}")

def get_bot_move(board: Board):
    """Get the best move from the bot given a board position"""
    global model, move_to_int, int_to_move
    
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Get legal moves first
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    
    # Convert board to model input format
    board_matrix = board_to_matrix(board)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(board_matrix, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Get legal move UCI strings
    legal_move_ucis = [move.uci() for move in legal_moves]
    
    # Filter predictions to only legal moves
    legal_move_scores = []
    for move_uci in legal_move_ucis:
        if move_uci in move_to_int:
            move_idx = move_to_int[move_uci]
            score = probabilities[0][move_idx].item()
            legal_move_scores.append((move_uci, score))
        else:
            # If move not in vocabulary, give it a very low score
            # This can happen if the model wasn't trained on all possible moves
            legal_move_scores.append((move_uci, 0.0))
    
    # Sort by score and get best move
    if not legal_move_scores:
        # Fallback: return random legal move
        import random
        return random.choice(legal_moves)
    
    legal_move_scores.sort(key=lambda x: x[1], reverse=True)
    best_move_uci = legal_move_scores[0][0]
    
    # Convert UCI string to Move object
    try:
        best_move = Move.from_uci(best_move_uci)
        # Double-check it's legal
        if best_move in legal_moves:
            return best_move
        else:
            # If somehow invalid, return first legal move
            return legal_moves[0]
    except Exception as e:
        # Fallback: return first legal move if parsing fails
        print(f"Error parsing move {best_move_uci}: {e}")
        return legal_moves[0]

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/move', methods=['POST'])
def make_move():
    """Handle a move from the player and return bot's move"""
    try:
        data = request.json
        fen = data.get('fen')
        move_uci = data.get('move')
        
        if not fen:
            return jsonify({'error': 'Missing FEN'}), 400
        
        # Create board from FEN (this should be the state AFTER player's move)
        board = Board(fen)
        
        # If move_uci is provided, validate it (for debugging)
        if move_uci:
            try:
                move = Move.from_uci(move_uci)
                # Check if this move would result in the current FEN
                # (This is just validation, the board is already in the post-move state)
            except:
                pass  # Ignore validation errors, trust the FEN
        
        # Check if game is over
        if board.is_checkmate():
            return jsonify({
                'bot_move': None,
                'fen': board.fen(),
                'game_over': True,
                'winner': 'player'
            })
        
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventy_five_moves() or board.is_fivefold_repetition():
            return jsonify({
                'bot_move': None,
                'fen': board.fen(),
                'game_over': True,
                'winner': 'draw'
            })
        
        # Get bot's move
        bot_move = get_bot_move(board)
        
        if bot_move is None:
            return jsonify({'error': 'No legal moves available'}), 400
        
        # Make bot's move
        board.push(bot_move)
        
        # Check if game is over after bot's move
        game_over = False
        winner = None
        
        if board.is_checkmate():
            game_over = True
            winner = 'bot'
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventy_five_moves() or board.is_fivefold_repetition():
            game_over = True
            winner = 'draw'
        
        return jsonify({
            'bot_move': bot_move.uci(),
            'fen': board.fen(),
            'game_over': game_over,
            'winner': winner
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game"""
    board = Board()
    return jsonify({
        'fen': board.fen(),
        'message': 'New game started'
    })

if __name__ == '__main__':
    print("Loading model...")
    try:
        load_model_and_vocab()
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPlease ensure you have:")
        print("1. Trained a model using train.py")
        print("2. Saved the model and vocabulary")
        print("\nYou can still run the server, but the bot won't work without a trained model.")

