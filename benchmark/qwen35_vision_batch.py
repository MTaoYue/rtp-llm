"""Run a custom uniform image batch with the standard vision benchmark."""

import argparse
import sys

import qwen35_vision as benchmark


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=448)
    args, remaining = parser.parse_known_args()
    if args.batch_size <= 0 or args.resolution <= 0 or args.resolution % 32:
        parser.error(
            "batch-size must be positive; resolution must be a positive multiple of 32"
        )
    grid = [(1, args.resolution // 16, args.resolution // 16)] * args.batch_size
    name = f"batch_{args.batch_size}_{args.resolution}"
    benchmark.cases = lambda: [(name, grid)]
    sys.argv = [sys.argv[0], *remaining]
    benchmark.main()


if __name__ == "__main__":
    main()
