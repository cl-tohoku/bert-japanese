# Copyright 2020 The HuggingFace Inc. team.
# Copyright 2021 Masatoshi Suzuki (@singletongue)
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

from logzero import logger

from japanese_tokenizers.implementations import JapaneseWordPieceTokenizer


def main(args):
    tokenizer = JapaneseWordPieceTokenizer(
        num_unused_tokens=args.num_unused_tokens,
        mecab_dic_type=args.mecab_dic_type,
    )
    speical_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    speical_tokens += ['<unused{}>'.format(i) for i in range(args.num_unused_tokens)]

    logger.info("Training the tokenizer")
    tokenizer.train(
        args.input_files,
        vocab_size=args.vocab_size,
        limit_alphabet=args.limit_alphabet,
        special_tokens=speical_tokens
    )

    logger.info("Saving the tokenizer to files")
    tokenizer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer_type", type=str, required=True)
    parser.add_argument("--mecab_dic_type", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--limit_alphabet", type=int)
    parser.add_argument("--num_unused_tokens", type=int, default=10)
    args = parser.parse_args()
    main(args)
