# Quick Reference - NNUE Chess Bot Training

## 🚀 Fast Track (5 Minutes to Start Training)

### 1. Setup Data Directory
```bash
cd ~/Desktop/SWE/Personal/ChessBot
mkdir -p data
```

### 2. Get Training Data (Choose One)

**Option A: Quick Test (Small Dataset)**
```bash
cd data
wget https://www.pgnmentor.com/files/WorldChampions.zip
unzip WorldChampions.zip
```

**Option B: Serious Training (Lichess Database)**
```bash
cd data
# Download one month (example: January 2024)
wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst

# Decompress
brew install zstd  # macOS (or: sudo apt install zstd for Linux)
zstd -d lichess_db_standard_rated_2024-01.pgn.zst
```

### 3. Filter Data (Optional but Recommended)
```bash
cd ../compact

# Filter to high-quality games
python3 filter_pgn.py ../data/lichess_db_standard_rated_2024-01.pgn \
                      ../data/elite_games.pgn \
                      --min-elo 2200 \
                      --max-games 50000
```

### 4. Train!
```bash
python3 nnue_train.py

# Wait 2-4 hours (CPU) or 30-60 min (GPU)
```

### 5. Play!
```bash
python3 nnue_play.py
```

---

## 📁 File Structure

```
ChessBot/
├── data/              ← PUT YOUR PGN FILES HERE
│   ├── games.pgn
│   └── ...
└── compact/
    ├── nnue_train.py  ← RUN THIS TO TRAIN
    └── nnue_play.py   ← RUN THIS TO PLAY
```

---

## 🔧 Common Commands

### Check if data is ready:
```bash
ls -lh ../data/*.pgn
```

### Test the bot (before training):
```bash
python3 nnue_test.py
```

### Check model size:
```bash
python3 size_analysis.py
```

### Play against trained bot:
```bash
python3 nnue_play.py
```

---

## 📊 Training Parameters (in nnue_train.py)

**For quick test:**
```python
MAX_GAMES_PER_FILE = 1000
MAX_FILES = 10
NUM_EPOCHS = 30
```

**For max strength:**
```python
MAX_GAMES_PER_FILE = 5000
MAX_FILES = 50
NUM_EPOCHS = 100
MIN_ELO = 2400
```

---

## 🌐 Data Sources

| Source | URL | Quality |
|--------|-----|---------|
| Lichess Elite | https://database.lichess.org/ | ⭐⭐⭐⭐⭐ Best |
| CCRL | https://computerchess.org.uk/ccrl/ | ⭐⭐⭐⭐ Great |
| PGN Mentor | https://www.pgnmentor.com/ | ⭐⭐⭐ Good |

---

## 💡 Quick Tips

✅ **Use only 2200+ ELO games**
✅ **Classical time control preferred**
✅ **50,000-100,000 games = optimal**
✅ **More data = stronger bot**

❌ **Don't use bullet/blitz games**
❌ **Don't use < 2000 ELO games**
❌ **Don't overtrain (watch val loss)**

---

## 🐛 Quick Fixes

**"No PGN files found"**
→ Put `.pgn` files in `../data/` directory

**"Out of memory"**
→ Reduce `BATCH_SIZE` to 512 or 256

**"Training too slow"**
→ Reduce `MAX_GAMES_PER_FILE` and `MAX_FILES`

**"Model plays poorly"**
→ Use higher ELO data (2400+) and more games

---

## 📈 Expected Strength

After training on **50,000 high-quality games**:

| Depth | Time | ELO |
|-------|------|-----|
| 3 | 0.1s | 1600 |
| 4 | 0.5s | 1800 |
| 5 | 2s | **2000** ⭐ |
| 6 | 8s | **2200** |
| 7 | 30s | **2400+** |

---

## ✅ Checklist

- [ ] Install: `pip install torch chess numpy tqdm`
- [ ] Create: `mkdir -p ../data`
- [ ] Download PGN files
- [ ] Move files to `../data/`
- [ ] Run: `python3 nnue_train.py`
- [ ] Test: `python3 nnue_play.py`
- [ ] Compete! 🏆
