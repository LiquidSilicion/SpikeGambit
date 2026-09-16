#!/usr/bin/env python3
"""
generate_tactical_dataset.py
----------------------------
Generates a dataset of chess positions (FEN) and labels them with tactical motifs
using geometric heuristics. This dataset will be used to train the SpikeGambit SCNN.

Labels:
0: Quiet (No immediate tactical threat)
1: Check (King is under attack)
2: Hanging Piece (Piece is attacked and undefended)
3: Tactical Threat (Fork, Winning Capture, or Pin pattern)
"""

import chess
import csv
import random
import numpy as np
from pathlib import Path

# === CONFIGURATION ===
NUM_POSITIONS = 5000       # Total positions to generate
RANDOM_SEED = 42           # For reproducibility
OUTPUT_CSV = Path(__file__).parent / "tactical_dataset.csv"

# Piece values for heuristic evaluation
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

def detect_tactical_label(board):
    """
    Analyzes the board from the perspective of the current turn's player.
    Returns a tactical label (0-3) based on geometric heuristics.
    """
    # 1. Check for Check
    if board.is_check():
        return 1
        
    current_turn = board.turn
    opponent_turn = not current_turn
    
    has_hanging_piece = False
    has_tactical_threat = False
    
    # 2. Analyze current turn's pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.color != current_turn:
            continue
            
        piece_val = PIECE_VALUES[piece.piece_type]
        
        # Is this piece attacked by the opponent?
        if board.is_attacked_by(opponent_turn, square):
            # Is it undefended? (Hanging piece)
            if not board.is_attacked_by(current_turn, square):
                has_hanging_piece = True
            else:
                # Are the attackers of lower value? (Winning capture / Pin setup)
                attackers = board.attackers(opponent_turn, square)
                for attacker_sq in attackers:
                    attacker_piece = board.piece_at(attacker_sq)
                    if attacker_piece and PIECE_VALUES[attacker_piece.piece_type] < piece_val:
                        has_tactical_threat = True
                        
        # 3. Check for Forks (Is this piece attacking multiple high-value enemies?)
        high_value_targets = 0
        for enemy_sq in chess.SQUARES:
            enemy_piece = board.piece_at(enemy_sq)
            if enemy_piece and enemy_piece.color == opponent_turn:
                if board.is_attacked_by(current_turn, enemy_sq):
                    # Target is valuable (>= attacker value) or is the King
                    if PIECE_VALUES[enemy_piece.piece_type] >= piece_val or enemy_piece.piece_type == chess.KING:
                        high_value_targets += 1
        
        if high_value_targets >= 2:
            has_tactical_threat = True

    # Priority: Tactical Threat > Hanging Piece > Quiet
    if has_tactical_threat:
        return 3
    if has_hanging_piece:
        return 2
    return 0


def fen_to_tensor(fen):
    """
    Converts a FEN string into a 12x8x8 numpy array (Channels x Height x Width).
    12 Channels: [White Pawn, White Knight, ..., Black King]
    Values: 1.0 if piece is present, 0.0 otherwise.
    This is the exact input format required for the SCNN.
    """
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    # Channel mapping: 0-5 for White (P, N, B, R, Q, K), 6-11 for Black
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
            row = 7 - chess.square_rank(square)  # Flip so row 0 is rank 8
            col = chess.square_file(square)
            tensor[channel, row, col] = 1.0
            
    return tensor


def generate_dataset():
    """Generates random mid-game positions and labels them."""
    print(f"🎲 Generating {NUM_POSITIONS} chess positions and labeling tactics...")
    random.seed(RANDOM_SEED)
    
    dataset = []
    attempts = 0
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    while len(dataset) < NUM_POSITIONS and attempts < NUM_POSITIONS * 5:
        board = chess.Board()
        # Play 10 to 25 random moves to reach a mid-game state
        num_moves = random.randint(10, 25)
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            board.push(random.choice(legal_moves))
        
        # Only keep valid mid-game positions (at least 16 pieces, not game over)
        if not board.is_game_over() and len(board.piece_map()) >= 16:
            fen = board.fen()
            label = detect_tactical_label(board)
            dataset.append((fen, label))
            label_counts[label] += 1
            
        attempts += 1
        
        if len(dataset) % 500 == 0 and len(dataset) > 0:
            print(f"   📊 Processed {len(dataset)} positions...")

    # Save to CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "tactical_label"])
        for fen, label in dataset:
            writer.writerow([fen, label])
            
    print(f"\n✅ Dataset saved to: {OUTPUT_CSV}")
    print(f"📈 Label Distribution:")
    print(f"   0 (Quiet):           {label_counts[0]} ({label_counts[0]/len(dataset)*100:.1f}%)")
    print(f"   1 (Check):           {label_counts[1]} ({label_counts[1]/len(dataset)*100:.1f}%)")
    print(f"   2 (Hanging Piece):   {label_counts[2]} ({label_counts[2]/len(dataset)*100:.1f}%)")
    print(f"   3 (Tactical Threat): {label_counts[3]} ({label_counts[3]/len(dataset)*100:.1f}%)")
    
    # Quick test of the tensor conversion
    print(f"\n🧪 Testing FEN to SCNN Tensor conversion...")
    sample_fen = dataset[0][0]
    tensor = fen_to_tensor(sample_fen)
    print(f"   Input FEN: {sample_fen}")
    print(f"   Output Tensor Shape: {tensor.shape} (Channels x 8x8 Board)")
    print(f"   Total pieces encoded: {int(tensor.sum())}")


if __name__ == "__main__":
    generate_dataset()