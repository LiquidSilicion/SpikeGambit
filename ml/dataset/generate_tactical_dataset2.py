#!/usr/bin/env python3
"""
generate_tactical_dataset_balanced.py
-------------------------------------
Generates a strictly balanced dataset (50/50) of chess positions labeled as 
Tactical or Quiet using Stockfish evaluation thresholds.
"""

import chess
import chess.engine
import csv
import random
from pathlib import Path

# === CONFIGURATION ===
NUM_POSITIONS = 4000  # Total positions (2000 Tactical, 2000 Quiet)
RANDOM_SEED = 42
OUTPUT_CSV = Path(__file__).parent / "tactical_dataset_balanced.csv"
STOCKFISH_PATH = "/usr/games/stockfish"  # Adjust if your stockfish is elsewhere

def get_stockfish_eval(board, engine, depth=10):
    """Returns the relative centipawn evaluation from Stockfish."""
    try:
        result = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = result['score'].relative
        if score.is_mate():
            return 10000 if score.relative.mate() > 0 else -10000
        return score.score()
    except Exception:
        return 0

def generate_balanced_dataset():
    print(f"🎲 Generating {NUM_POSITIONS} balanced chess positions (50% Tactical, 50% Quiet)...")
    random.seed(RANDOM_SEED)
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        print("✅ Stockfish initialized")
    except Exception as e:
        print(f"❌ Stockfish not found at {STOCKFISH_PATH}. Please install it: sudo apt install stockfish")
        return

    dataset = []
    tactical_count = 0
    quiet_count = 0
    attempts = 0
    
    target_per_class = NUM_POSITIONS // 2
    
    print("Starting generation. This may take a few minutes as it filters ambiguous positions...")
    
    while (tactical_count < target_per_class or quiet_count < target_per_class) and attempts < 100000:
        board = chess.Board()
        
        # Play 5 to 12 random moves to get out of the opening, but avoid total chaos
        num_moves = random.randint(5, 12)
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            board.push(random.choice(legal_moves))
            
        if board.is_game_over() or len(board.piece_map()) < 10:
            attempts += 1
            continue
            
        fen = board.fen()
        eval_cp = get_stockfish_eval(board, engine, depth=10)
        is_check = board.is_check()
        
        # Determine label based on strict thresholds
        if is_check or abs(eval_cp) > 150:
            label = 1  # Tactical
        elif abs(eval_cp) < 50 and not is_check:
            label = 0  # Quiet
        else:
            # Ambiguous position (0.5 to 1.5 pawn advantage), skip it
            attempts += 1
            continue
            
        # Enforce strict 50/50 balance
        if label == 1 and tactical_count < target_per_class:
            dataset.append((fen, label))
            tactical_count += 1
        elif label == 0 and quiet_count < target_per_class:
            dataset.append((fen, label))
            quiet_count += 1
            
        attempts += 1
        
        if (tactical_count + quiet_count) % 500 == 0 and (tactical_count + quiet_count) > 0:
            print(f"   📊 Progress: {tactical_count + quiet_count}/{NUM_POSITIONS} (Tactical: {tactical_count}, Quiet: {quiet_count})")

    engine.quit()
    
    # Shuffle the dataset so tactical and quiet are mixed
    random.shuffle(dataset)
    
    # Save to CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "is_tactical"])
        for fen, label in dataset:
            writer.writerow([fen, label])
            
    print(f"\n✅ Dataset saved to: {OUTPUT_CSV}")
    print(f"📈 Final Label Distribution:")
    print(f"   0 (Quiet):      {quiet_count} ({quiet_count/len(dataset)*100:.1f}%)")
    print(f"   1 (Tactical):   {tactical_count} ({tactical_count/len(dataset)*100:.1f}%)")

if __name__ == "__main__":
    generate_balanced_dataset()