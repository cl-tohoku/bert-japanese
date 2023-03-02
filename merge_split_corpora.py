# Copyright 2023 Masatoshi Suzuki (@singletongue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gzip
import logging
import lzma
import os
import random

from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(0)


def _open_file(filename):
    if filename.endswith(".xz"):
        return lzma.open(filename, "rt")
    elif filename.endswith(".gz"):
        return gzip.open(filename, "rt")
    else:
        return open(filename)


def main(args):
    output_files = []
    for i in range(1, args.num_files + 1):
        output_path = os.path.join(args.output_dir, f"corpus_{i:02d}.txt")
        output_file = open(output_path, "w")
        output_files.append(output_file)

    output_index = random.randint(1, args.num_files)

    for input_path in args.input_files:
        logger.info("Processing %s", input_path)
        with _open_file(input_path) as f:
            for line in tqdm(f):
                line = " ".join(line.strip().split())
                print(line, file=output_files[output_index])

                if line == "":
                    output_index = random.randrange(args.num_files)

    for output_file in output_files:
        output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_files", type=int, required=True)
    args = parser.parse_args()
    main(args)
