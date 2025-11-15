from flask import Flask, render_template, request, jsonify
import os
import sys
from chess import Board, Move

# Add compact directory to path to import NNUE engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'compact'))

from nnue_engine import NNUEEngine
from opening_book import OpeningBook

app = Flask(__name__)

# Global variables for engine
engine = None
opening_book = None
search_depth = 4  # Default search depth

def load_engine():
    """Load the NNUE engine and opening book"""
    global engine, opening_book
    
    # Try to find NNUE model files
    model_paths = [
        "compact/nnue_best.pth",
        "compact/nnue_final.pth",
        "compact/nnue_model.pth",
        "nnue_best.pth",
        "nnue_final.pth",
        "nnue_model.pth",
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError(
            "No NNUE model found. Please train a model first.\n"
            "Expected files: nnue_best.pth, nnue_final.pth, or nnue_model.pth"
        )
    
    print(f"Loading NNUE engine from: {model_path}")
    
    # Initialize engine with smaller TT for web server
    engine = NNUEEngine(model_path, device="cpu", tt_size_mb=2.2)
    print("✓ NNUE engine loaded!")
    
    # Load opening book (optional)
    opening_book = None
    try:
        opening_book = OpeningBook()
        print(f"✓ Opening book loaded ({len(opening_book.book)} positions)")
    except Exception as e:
        print(f"⚠ Warning: Could not load opening book: {e}")
        print("   Continuing without opening book...")
        opening_book = None

def get_bot_move(board: Board, depth: int = None):
    """Get the best move from the bot given a board position"""
    global engine, opening_book
    
    if engine is None:
        raise RuntimeError("Engine not loaded. Please train a model first.")
    
    # Use provided depth or default
    if depth is None:
        depth = search_depth
    
    # Check opening book first
    if opening_book and opening_book.is_in_book(board):
        book_move = opening_book.get_book_move(board)
        if book_move:
            return book_move
    
    # Use engine to find best move
    # Suppress print output from engine during web requests
    import io
    import contextlib
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        best_move = engine.get_move(board, depth=depth)
    
    return best_move

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
        depth = data.get('depth', search_depth)  # Allow depth to be specified
        
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
        bot_move = get_bot_move(board, depth=depth)
        
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
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game"""
    board = Board()
    return jsonify({
        'fen': board.fen(),
        'message': 'New game started'
    })

@app.route('/api/set_depth', methods=['POST'])
def set_depth():
    """Set the search depth for the engine"""
    global search_depth
    try:
        data = request.json
        depth = data.get('depth', 4)
        if 1 <= depth <= 7:
            search_depth = depth
            return jsonify({'success': True, 'depth': search_depth})
        else:
            return jsonify({'error': 'Depth must be between 1 and 7'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Loading NNUE Chess Engine...")
    print("=" * 60)
    try:
        load_engine()
        print("\n" + "=" * 60)
        print("Starting Flask server...")
        print("=" * 60)
        print("Open your browser to: http://localhost:5000")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n❌ Error loading engine: {e}")
        print("\nPlease ensure you have:")
        print("1. Trained an NNUE model (nnue_best.pth, nnue_final.pth, or nnue_model.pth)")
        print("2. The model file is in the 'compact' directory or root directory")
        print("\n⚠️  Starting server anyway, but the bot won't work without a trained model.")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("Starting Flask server (without engine)...")
        print("=" * 60)
        print("Open your browser to: http://localhost:5000")
        print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

