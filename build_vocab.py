import tempfile
import os
import argparse

import sentencepiece
import tensorflow as tf

from tokenization import MecabBasicTokenizer


CONTROL_SYMBOLS = ['[CLS]', '[SEP]', '[MASK]']


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    tokenizer = MecabBasicTokenizer(do_lower_case=args.do_lower_case,
                                    dict_path=args.mecab_dict)
    with tempfile.TemporaryDirectory() as tempdir:
        # read input files and write to a temporary file
        concat_input_file = open(os.path.join(tempdir, 'input.txt'), 'w')
        for input_file in tf.gfile.Glob(args.input_file):
            with tf.gfile.GFile(input_file, 'r') as reader:
                tf.logging.info('Reading {}'.format(input_file))
                for line in reader:
                    tokens = tokenizer.tokenize(line.strip('\n'))
                    print(' '.join(tokens), file=concat_input_file)

        # train a SentencePiece model and store the vocabulary file to a temp directory
        tf.logging.info('Training a SentencePiece model')
        commands = {
            'input': concat_input_file.name,
            'model_prefix': os.path.join(tempdir, 'sp'),
            'model_type': args.vocab_type,
            'normalization_rule_name': 'identity',
            'vocab_size': args.vocab_size,
            'pad_id': 0,
            'unk_id': 1,
            'bos_id': -1,
            'eos_id': -1,
            'control_symbols': ','.join(CONTROL_SYMBOLS),
            'input_sentence_size': args.sentence_size,
            'shuffle_input_sentence': 'true'
        }
        command_line = ' '.join(['--{}={}'.format(k, v) for k, v in commands.items()])
        sentencepiece.SentencePieceTrainer.Train(command_line)
        concat_input_file.close()

        # convert SentencePiece vocabulary into WordPiece format that is used in BERT
        with open(os.path.join(tempdir, 'sp.vocab')) as vocab_file, \
             tf.gfile.GFile(args.output_file, 'w') as output_file:
            for line in vocab_file:
                sp_token, _ = line.rstrip('\n').split('\t')
                if sp_token == '<pad>':
                    wp_token = '[PAD]'
                elif sp_token == '<unk>':
                    wp_token = '[UNK]'
                elif sp_token in CONTROL_SYMBOLS:
                    wp_token = sp_token
                elif sp_token.startswith('\u2581'):
                    # e.g. "▁word" -> "word"
                    wp_token = sp_token[1:]
                elif args.vocab_type == 'bpe':
                    # e.g. "tion" -> "##tion"
                    wp_token = '##' + sp_token
                else:
                    wp_token = sp_token

                output_file.write(wp_token + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
        help='Input raw text file (or comma-separated list of files).')
    parser.add_argument('--output_file', type=str, required=True,
        help='Output vocabulary file.')
    parser.add_argument('--vocab_type', choices=('bpe', 'char'), default='bpe',
        help='Subword vocabulary type ("bpe" or "char"). [bpe]')
    parser.add_argument('--vocab_size', type=int, default=32000,
        help='WordPiece vocabulary size. [32000]')
    parser.add_argument('--sentence_size', type=int, default=1000000,
        help='Limit the input sentence size. [1000000]')
    parser.add_argument('--do_lower_case', type=bool, default=False,
        help='Whether to lower case the input text. [False]')
    parser.add_argument('--mecab_dict', type=str,
        help='Path to MeCab custom dictionary.')
    args = parser.parse_args()

    main(args)
