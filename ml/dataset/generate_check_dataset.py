#!/usr/bin/env python3
"""
generate_check_dataset.py
Generates a dataset specifically for detecting if the king is in check.
"""

import chess
import csv
import random
from pathlib import Path
import numpy as np

NUM_POSITIONS = 5000
RANDOM_SEED = 42
OUTPUT_CSV = Path(__file__).parent / "check_dataset.csv"

def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            tensor[channel, row, col] = 1.0
            
    return tensor

def generate_dataset():
    print(f"🎲 Generating {NUM_POSITIONS} positions for check detection...")
    random.seed(RANDOM_SEED)
    
    dataset = []
    check_count = 0
    no_check_count = 0
    attempts = 0
    
    target_per_class = NUM_POSITIONS // 2
    
    while (check_count < target_per_class or no_check_count < target_per_class) and attempts < 50000:
        board = chess.Board()
        num_moves = random.randint(5, 20)
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            board.push(random.choice(legal_moves))
        
        if board.is_game_over() or len(board.piece_map()) < 10:
            attempts += 1
            continue
        
        is_check = board.is_check()
        label = 1 if is_check else 0
        
        if label == 1 and check_count < target_per_class:
            dataset.append((board.fen(), label))
            check_count += 1
        elif label == 0 and no_check_count < target_per_class:
            dataset.append((board.fen(), label))
            no_check_count += 1
        
        attempts += 1
        
        if (check_count + no_check_count) % 500 == 0 and (check_count + no_check_count) > 0:
            print(f"   📊 Progress: {check_count + no_check_count}/{NUM_POSITIONS} (Check: {check_count}, No Check: {no_check_count})")
    
    random.shuffle(dataset)
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "is_check"])
        for fen, label in dataset:
            writer.writerow([fen, label])
    
    print(f"\n✅ Dataset saved to: {OUTPUT_CSV}")
    print(f"📈 Final Distribution:")
    print(f"   0 (No Check): {no_check_count} ({no_check_count/len(dataset)*100:.1f}%)")
    print(f"   1 (Check):    {check_count} ({check_count/len(dataset)*100:.1f}%)")

if __name__ == "__main__":
    generate_dataset()