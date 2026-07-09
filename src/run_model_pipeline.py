from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "common"))

from config import TARGET_ORDER  # noqa: E402
from pipeline import run_target_step  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all endpoint models with checkpointing.")
    parser.add_argument("--time-budget", type=int, default=120, help="Seconds per endpoint step.")
    parser.add_argument("--max-cycles", type=int, default=20, help="Maximum passes across endpoints.")
    args = parser.parse_args()

    remaining = set(TARGET_ORDER)
    for cycle in range(1, args.max_cycles + 1):
        if not remaining:
            break
        print(f"\n=== Training cycle {cycle}; remaining endpoints: {len(remaining)} ===", flush=True)
        for target in list(TARGET_ORDER):
            if target not in remaining:
                continue
            done = run_target_step(target, time_budget=args.time_budget)
            if done:
                remaining.remove(target)

    if remaining:
        names = ", ".join(sorted(remaining))
        raise SystemExit(f"Training incomplete after {args.max_cycles} cycles: {names}")

    print("All endpoint models completed.")


if __name__ == "__main__":
    main()
