#!/bin/bash
# Complete Modal training workflow

echo "============================================================"
echo " NNUE CHESS BOT - MODAL GPU TRAINING"
echo "============================================================"
echo ""

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal not installed"
    echo ""
    echo "Install with:"
    echo "  pip install modal"
    echo ""
    exit 1
fi

# # Check if authenticated
# if ! modal token show &> /dev/null; then
#     echo "❌ Not authenticated with Modal"
#     echo ""
#     echo "Run:"
#     echo "  modal setup"
#     echo ""
#     exit 1
# fi

echo "✓ Modal installed and authenticated"
echo ""

# Check if data exists
if [ ! -d "../data" ]; then
    echo "❌ Data directory not found: ../data"
    exit 1
fi

PGN_COUNT=$(ls -1 ../data/*.pgn 2>/dev/null | wc -l | tr -d ' ')
if [ "$PGN_COUNT" -eq 0 ]; then
    echo "❌ No PGN files found in ../data/"
    exit 1
fi

echo "✓ Found $PGN_COUNT PGN files"
echo ""

# Show file sizes
echo "Your data files:"
ls -lh ../data/*.pgn | awk '{print "  " $9 ": " $5}'
echo ""

# Check if data is already uploaded
echo "Checking Modal storage..."
UPLOADED=$(modal volume ls chess-data 2>/dev/null | grep -c ".pgn" || echo "0")

if [ "$UPLOADED" -gt 0 ]; then
    echo "✓ Data already uploaded to Modal ($UPLOADED files)"
    echo ""
    read -p "Re-upload data? (y/n) [n]: " REUPLOAD
    REUPLOAD=${REUPLOAD:-n}
else
    echo "⚠️  Data not yet uploaded to Modal"
    REUPLOAD="y"
fi

# Upload data if needed
if [ "$REUPLOAD" = "y" ]; then
    echo ""
    echo "============================================================"
    echo " STEP 1: UPLOADING DATA TO MODAL"
    echo "============================================================"
    echo ""
    echo "This will upload your PGN files to Modal storage..."
    echo "This is a one-time upload (~5-10 minutes for 2.2 GB)"
    echo ""

    read -p "Continue? (y/n): " CONFIRM
    if [ "$CONFIRM" != "y" ]; then
        echo "Cancelled"
        exit 0
    fi

    echo ""
    echo "Creating Modal volume..."
    modal volume create chess-data 2>/dev/null || echo "Volume already exists"

    echo ""
    echo "Uploading files (this may take 5-10 minutes)..."
    echo ""

    # Upload each file
    for file in ../data/*.pgn; do
        filename=$(basename "$file")
        echo "Uploading $filename..."
        modal volume put chess-data "$file" "/" || {
            echo "❌ Upload failed!"
            exit 1
        }
    done

    echo ""
    echo "✓ Data uploaded successfully!"
    echo ""
fi

# Train
echo "============================================================"
echo " STEP 2: TRAINING ON GPU"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  • GPU: T4 (NVIDIA)"
echo "  • Games: ~40,000 (2200+ ELO)"
echo "  • Epochs: 50"
echo "  • Time: 30-60 minutes"
echo "  • Cost: ~\$0.50-1.00"
echo ""

read -p "Start training? (y/n): " TRAIN
if [ "$TRAIN" != "y" ]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "🚀 Starting training on Modal GPU..."
echo ""

# Run training
modal run train_modal_fixed.py || {
    echo ""
    echo "❌ Training failed!"
    exit 1
}

echo ""
echo "============================================================"
echo " STEP 3: DOWNLOAD MODEL"
echo "============================================================"
echo ""

read -p "Download trained model? (y/n) [y]: " DOWNLOAD
DOWNLOAD=${DOWNLOAD:-y}

if [ "$DOWNLOAD" = "y" ]; then
    echo ""
    echo "Downloading nnue_best.pth..."
    modal volume get chess-models nnue_best.pth . || {
        echo "❌ Download failed!"
        exit 1
    }

    echo ""
    echo "✓ Model downloaded!"

    # Check size
    SIZE=$(ls -lh nnue_best.pth | awk '{print $5}')
    echo "✓ Size: $SIZE"
    echo ""
fi

echo "============================================================"
echo " COMPLETE! 🎉"
echo "============================================================"
echo ""
echo "Your chess bot is trained and ready!"
echo ""
echo "Next steps:"
echo "  1. Verify size:  python3 size_analysis.py"
echo "  2. Test it:      python3 nnue_test.py"
echo "  3. Play:         python3 nnue_play.py"
echo ""
echo "Enjoy your Master-level chess bot! 🏆"
echo ""
