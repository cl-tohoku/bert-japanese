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
import csv
from unicodedata import normalize


def main(args):
    seen_chars = set()
    with open(args.lex_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            token = row[0]
            if token == "":
                token = '"'

            for char in list(token):
                char = normalize("NFKC", char)
                if len(char) != 1:
                    continue

                if not char.isprintable():
                    continue

                seen_chars.add(char)

    with open(args.output_file, "w") as fo:
        for char in sorted(list(seen_chars)):
            print(char, file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lex_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
