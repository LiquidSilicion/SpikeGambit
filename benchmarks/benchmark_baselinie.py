#!/usr/bin/env python3
"""
benchmark_baseline.py — Standalone IEEE Paper Baseline
Uses python-chess minimax (no custom engine dependencies)
"""

import time
import csv
import random
import statistics
from pathlib import Path
import chess

# === CONFIGURATION ===
SEARCH_DEPTH = 3
NUM_POSITIONS = 100
RANDOM_SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"
POSITIONS_FILE = Path(__file__).parent / "test_positions.fen"


def load_or_generate_positions():
    """Load or generate chess positions robustly."""
    positions = []
    if POSITIONS_FILE.exists():
        with open(POSITIONS_FILE, 'r') as f:
            positions = [line.strip() for line in f if line.strip()]
        
        if len(positions) >= NUM_POSITIONS:
            print(f"📂 Loaded {NUM_POSITIONS} positions from {POSITIONS_FILE.name}")
            return positions[:NUM_POSITIONS]
        else:
            print(f"⚠️ {POSITIONS_FILE.name} only has {len(positions)} valid positions. Regenerating...")
            POSITIONS_FILE.unlink()  # Delete the bad/empty file

    print(f"🎲 Generating {NUM_POSITIONS} random mid-game positions (seed={RANDOM_SEED})...")
    random.seed(RANDOM_SEED)
    positions = []
    attempts = 0
    
    while len(positions) < NUM_POSITIONS and attempts < 1000:
        board = chess.Board()
        num_moves = random.randint(10, 25)
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            board.push(random.choice(legal_moves))
        
        # Ensure it's a valid mid-game position with enough pieces
        if board.piece_map() and len(board.piece_map()) >= 8 and not board.is_game_over():
            positions.append(board.fen())
        attempts += 1
    
    POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POSITIONS_FILE, 'w') as f:
        for fen in positions:
            f.write(fen + '\n')
    print(f"💾 Saved {len(positions)} positions to {POSITIONS_FILE.name}")
    return positions


def simple_eval(board):
    """Simple material evaluation."""
    PIECE_VALUES = {'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000}
    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES.get(piece.symbol().lower(), 0)
        score += value if piece.color == chess.WHITE else -value
    return score


def minimax(board, depth, alpha, beta, maximizing):
    """Simple minimax with alpha-beta pruning."""
    if depth == 0 or board.is_game_over():
        return simple_eval(board)
    
    legal_moves = list(board.legal_moves)
    if maximizing:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval_ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_)
            alpha = max(alpha, eval_)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_)
            beta = min(beta, eval_)
            if beta <= alpha:
                break
        return min_eval


def run_benchmark():
    """Execute the benchmark."""
    positions = load_or_generate_positions()
    
    if not positions:
        print("❌ Error: No positions were generated or loaded. Exiting.")
        return

    print(f"\n{'='*60}")
    print(f"🏁 BASELINE BENCHMARK — SpikeGambit IEEE Paper")
    print(f"{'='*60}")
    print(f"Search Depth:    {SEARCH_DEPTH}")
    print(f"Positions:       {len(positions)}")
    print(f"{'='*60}\n")
    
    times_ms = []
    benchmark_start = time.perf_counter()
    
    for i, fen in enumerate(positions, 1):
        board = chess.Board(fen)
        start_time = time.perf_counter()
        
        # Find best move using minimax
        best_move = None
        best_eval = float('-inf') if board.turn == chess.WHITE else float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            eval_ = minimax(board, SEARCH_DEPTH - 1, float('-inf'), float('inf'), 
                          board.turn != chess.WHITE)
            board.pop()
            
            if board.turn == chess.WHITE:
                if eval_ > best_eval:
                    best_eval = eval_
                    best_move = move
            else:
                if eval_ < best_eval:
                    best_eval = eval_
                    best_move = move
        
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        times_ms.append(elapsed_ms)
        
        if i % 10 == 0 or i == 1:
            print(f"  [{i:3d}/{len(positions)}] {elapsed_ms:7.2f} ms | {best_move}")
    
    benchmark_end = time.perf_counter()
    total_time_s = benchmark_end - benchmark_start
    
    # Compute statistics
    avg_time_ms = statistics.mean(times_ms)
    median_time_ms = statistics.median(times_ms)
    p95_time_ms = sorted(times_ms)[int(0.95 * len(times_ms))]
    avg_pos_per_sec = 1000 / avg_time_ms if avg_time_ms > 0 else 0
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Total Time:          {total_time_s:.2f} s")
    print(f"Avg Time/Move:       {avg_time_ms:.2f} ms")
    print(f"Median Time/Move:    {median_time_ms:.2f} ms")
    print(f"95th Percentile:     {p95_time_ms:.2f} ms")
    print(f"Positions/Second:    {avg_pos_per_sec:.2f}")
    print(f"{'='*60}\n")
    
    # Save to CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "baseline_metrics.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "unit"])
        writer.writerow(["search_depth", SEARCH_DEPTH, "plies"])
        writer.writerow(["num_positions", len(positions), "count"])
        writer.writerow(["total_time", f"{total_time_s:.3f}", "seconds"])
        writer.writerow(["avg_time_per_move", f"{avg_time_ms:.3f}", "ms"])
        writer.writerow(["median_time_per_move", f"{median_time_ms:.3f}", "ms"])
        writer.writerow(["p95_time_per_move", f"{p95_time_ms:.3f}", "ms"])
        writer.writerow(["positions_per_second", f"{avg_pos_per_sec:.2f}", "pos/s"])
        writer.writerow(["baseline_type", "python-chess-minimax", "string"])
    
    print(f"💾 Results saved to: {csv_path}")
    print(f"\n✅ Benchmark complete. Use these numbers for IEEE Table 1.")


if __name__ == "__main__":
    run_benchmark()