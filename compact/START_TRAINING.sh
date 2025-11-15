#!/bin/bash
# Quick script to start training with your data

echo "============================================================"
echo "NNUE CHESS BOT - TRAINING"
echo "============================================================"
echo ""
echo "Your data:"
echo "  Files: 4 PGN files"
echo "  Games: ~400,000 elite games"
echo "  Size: 2.2 GB"
echo ""
echo "Training options:"
echo ""
echo "1. QUICK TRAINING (1-2 hours)"
echo "   - 8,000 games (~200K positions)"
echo "   - Good strength: ~2000 ELO"
echo ""
echo "2. STRONG TRAINING (3-4 hours)"
echo "   - 40,000 games (~1M positions)"
echo "   - Better strength: ~2100-2200 ELO"
echo ""
echo "3. MAX STRENGTH (6-8 hours)"
echo "   - 100,000 games (~2.5M positions)"
echo "   - Best strength: ~2200-2400 ELO"
echo ""

read -p "Choose option (1/2/3) [default: 1]: " choice

# Set defaults
choice=${choice:-1}

case $choice in
    1)
        echo ""
        echo "Starting QUICK TRAINING..."
        echo "  Games: 8,000"
        echo "  Time: 1-2 hours"
        echo ""
        python3 nnue_train.py
        ;;
    2)
        echo ""
        echo "Starting STRONG TRAINING..."
        echo "  Games: 40,000"
        echo "  Time: 3-4 hours"
        echo ""
        # Temporarily modify settings
        python3 -c "
import sys
sys.path.insert(0, '.')
from nnue_train import *

# Override settings
MAX_GAMES_PER_FILE = 10000
MAX_FILES = 4
NUM_EPOCHS = 70

# Run training
main()
" 2>&1 | tee training_log.txt
        ;;
    3)
        echo ""
        echo "Starting MAX STRENGTH TRAINING..."
        echo "  Games: 100,000"
        echo "  Time: 6-8 hours"
        echo ""
        echo "WARNING: This will take a long time!"
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            # Temporarily modify settings
            python3 -c "
import sys
sys.path.insert(0, '.')
from nnue_train import *

# Override settings
MAX_GAMES_PER_FILE = 25000
MAX_FILES = 4
NUM_EPOCHS = 100
POSITIONS_PER_GAME = 30

# Run training
main()
" 2>&1 | tee training_log.txt
        else
            echo "Cancelled."
        fi
        ;;
    *)
        echo "Invalid choice. Using default (option 1)..."
        python3 nnue_train.py
        ;;
esac

echo ""
echo "============================================================"
echo "TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "Your trained model is ready:"
echo "  nnue_best.pth - Best model from training"
echo "  nnue_final.pth - Final model after all epochs"
echo ""
echo "Next steps:"
echo "  1. Verify size:  python3 size_analysis.py"
echo "  2. Test it:      python3 nnue_test.py"
echo "  3. Play:         python3 nnue_play.py"
echo ""
