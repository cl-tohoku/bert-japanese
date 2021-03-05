import argparse
import os
import random

from logzero import logger
from tqdm import tqdm


random.seed(0)


def main(args):
    output_files = []
    for i in range(1, args.num_files + 1):
        output_path = os.path.join(args.output_dir, f"corpus_{i:02d}.txt")
        output_file = open(output_path, "w")
        output_files.append(output_file)

    output_index = random.randint(1, args.num_files)

    for input_path in args.input_files:
        logger.info(f"Reading {input_path}")
        with open(input_path, "r") as f:
            for line in tqdm(f):
                line = line.rstrip("\n")
                print(line, file=output_files[output_index])

                if line == "":
                    output_index = random.randrange(args.num_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_files", type=int, required=True)
    args = parser.parse_args()
    main(args)
