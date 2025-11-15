let board = null;
let game = new Chess();
let boardFlipped = false;
let playerColor = 'w'; // Player is white
let botColor = 'b'; // Bot is black
let isPlayerTurn = true;
let moveHistory = [];

// Initialize the board
function initBoard() {
    const config = {
        position: 'start',
        draggable: true,
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd,
        pieceTheme: 'https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/img/chesspieces/wikipedia/{piece}.png'
    };
    
    board = Chessboard('board', config);
    updateStatus();
}

function onDragStart(source, piece, position, orientation) {
    // Don't allow dragging if it's not player's turn
    if (!isPlayerTurn) {
        return false;
    }
    
    // Don't allow dragging opponent's pieces
    if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return false;
    }
}

function onDrop(source, target) {
    // Don't allow moves if it's not player's turn
    if (!isPlayerTurn) {
        return false;
    }
    
    // Try to make the move
    const move = game.move({
        from: source,
        to: target,
        promotion: 'q' // Auto-promote to queen
    });
    
    // If the move is illegal, snap back
    if (move === null) {
        return 'snapback';
    }
    
    // Update move history
    addMoveToHistory(move);
    
    // Check if game is over
    if (game.isGameOver()) {
        handleGameOver();
        return;
    }
    
    // It's now the bot's turn
    isPlayerTurn = false;
    updateStatus('Bot is thinking...');
    
    // Send move to server and get bot's response
    makeBotMove();
}

function onSnapEnd() {
    board.position(game.fen());
}

function makeBotMove() {
    const fen = game.fen();
    const lastMove = game.history({ verbose: true }).pop();
    const moveUci = lastMove ? `${lastMove.from}${lastMove.to}` : null;
    
    // Show loading indicator
    updateStatus('Bot is thinking... <span class="loading"></span>');
    
    fetch('/api/move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            fen: fen,
            move: moveUci
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error:', data.error);
            updateStatus('Error: ' + data.error);
            isPlayerTurn = true; // Allow player to try again
            return;
        }
        
        // Update game state with bot's move
        if (data.bot_move) {
            const botMove = game.move({
                from: data.bot_move.substring(0, 2),
                to: data.bot_move.substring(2, 4),
                promotion: data.bot_move.length > 4 ? data.bot_move[4] : 'q'
            });
            
            if (botMove) {
                addMoveToHistory(botMove);
                board.position(game.fen());
            }
        }
        
        // Update game state from server
        if (data.fen) {
            game.load(data.fen);
            board.position(game.fen());
        }
        
        // Check if game is over
        if (data.game_over) {
            handleGameOver(data.winner);
        } else {
            // It's now player's turn
            isPlayerTurn = true;
            updateStatus();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        updateStatus('Error communicating with server');
        isPlayerTurn = true; // Allow player to try again
    });
}

function addMoveToHistory(move) {
    const moveNumber = Math.floor(game.history().length / 2) + 1;
    const isWhite = move.color === 'w';
    
    if (isWhite) {
        moveHistory.push({
            number: moveNumber,
            white: move.san,
            black: null
        });
    } else {
        if (moveHistory.length > 0) {
            moveHistory[moveHistory.length - 1].black = move.san;
        }
    }
    
    updateMoveHistory();
}

function updateMoveHistory() {
    const historyDiv = document.getElementById('move-history');
    historyDiv.innerHTML = '';
    
    moveHistory.forEach(move => {
        const moveDiv = document.createElement('div');
        moveDiv.innerHTML = `<strong>${move.number}.</strong> ${move.white || '...'} ${move.black ? move.black : ''}`;
        historyDiv.appendChild(moveDiv);
    });
    
    // Scroll to bottom
    historyDiv.scrollTop = historyDiv.scrollHeight;
}

function updateStatus(message) {
    const statusDiv = document.getElementById('status');
    
    if (message) {
        statusDiv.innerHTML = message;
        return;
    }
    
    if (game.isCheckmate()) {
        statusDiv.textContent = 'Checkmate! ' + (game.turn() === 'w' ? 'Black wins!' : 'White wins!');
    } else if (game.isDraw()) {
        statusDiv.textContent = 'Draw!';
    } else if (game.isCheck()) {
        statusDiv.textContent = 'Check! ' + (isPlayerTurn ? 'Your turn' : "Bot's turn");
    } else {
        statusDiv.textContent = isPlayerTurn ? 'Your turn (White)' : "Bot's turn (Black)";
    }
}

function handleGameOver(winner) {
    isPlayerTurn = false;
    
    let title = 'Game Over';
    let message = '';
    
    if (winner === 'player') {
        title = '🎉 You Win!';
        message = 'Congratulations! You checkmated the bot!';
    } else if (winner === 'bot') {
        title = '🤖 Bot Wins';
        message = 'The bot checkmated you. Better luck next time!';
    } else {
        title = 'Draw';
        message = 'The game ended in a draw.';
    }
    
    document.getElementById('game-over-title').textContent = title;
    document.getElementById('game-over-message').textContent = message;
    document.getElementById('game-over-modal').style.display = 'block';
}

function startNewGame() {
    game = new Chess();
    moveHistory = [];
    isPlayerTurn = true;
    board.position('start');
    updateStatus();
    document.getElementById('game-over-modal').style.display = 'none';
}

function flipBoard() {
    boardFlipped = !boardFlipped;
    board.flip();
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    initBoard();
    
    document.getElementById('new-game-btn').addEventListener('click', startNewGame);
    document.getElementById('flip-board-btn').addEventListener('click', flipBoard);
    document.getElementById('play-again-btn').addEventListener('click', startNewGame);
    
    // Close modal when clicking outside
    window.addEventListener('click', function(event) {
        const modal = document.getElementById('game-over-modal');
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
});

