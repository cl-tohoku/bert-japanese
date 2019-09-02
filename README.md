# BERT models pretrained on Japanese text

This repository contains scripts for training BERT models on Japnese text.

## Pretrained models

See [Releases](https://github.com/singletongue/japanese-bert/releases).

## Usage

Refer to `masked_lm_example.ipynb`.

---

## Logs of pretraining

### Process a Wikipedia dump file

```sh
$ python3 wikiextractor/WikiExtractor.py --output /work/m-suzuki/work/japanese-bert/jawiki-20190701/corpus --bytes 512M --compress --json --links --namespaces 0
--no_templates --min_text_length 16 --processes 20 /work/m-suzuki/data/wikipedia/dumps/jawiki/20190701/jawiki-20190701-pages-articles.xml.bz2
...
INFO: Finished 20-process extraction of 1157858 articles in 1955.2s (592.2 art/s)
INFO: total of page: 1667236, total of articl page: 1157858; total of used articl page: 1157858
```

### Generate corpus

```sh
$ seq -f %02g 0 8|xargs -L 1 -I {} -P 9 python3 make_corpus.py --input_file /work/m-suzuki/work/japanese-bert/jawiki-20190701/corpus/AA/wiki_{}.bz2 --output_file /work/m-suzuki/work/japanese-bert/jawiki-20190701/corpus/corpus.txt.{} --mecab_dict_path ~/local/lib/mecab/dic/ipadic-neologd-20180828
```

### Build vocabulary

```sh
$ python3 build_vocab.py --input_file '/work/m-suzuki/work/japanese-bert/jawiki-20190701/corpus/corpus.txt.*' --output_file /work/m-suzuki/work/japanese-bert/jawiki-20190701/mecab-ipadic-bpe-32k/vocab.txt --subword_type bpe --vocab_size 32000
$ cat bert_config_base.json |jq '.vocab_size=32000' > /work/m-suzuki/work/japanese-bert/jawiki-20190701/mecab-ipadic-bpe-32k/bert_config.json
```

```sh
$ python3 build_vocab.py --input_file '/work/m-suzuki/work/japanese-bert/jawiki-20190701/corpus/corpus.txt.*' --output_file /work/m-suzuki/work/japanese-bert/jawiki-20190701/mecab-ipadic-char-4k/vocab.txt --subword_type char --vocab_size 4000
$ cat bert_config_base.json |jq '.vocab_size=4000' > /work/m-suzuki/work/japanese-bert/jawiki-20190701/mecab-ipadic-char-4k/bert_config.json
```

### Create pretraining data

```sh
$ seq -f %02g 0 8|xargs -L 1 -I {} -P 3 python3 create_pretraining_data.py --input_file /work/m-suzuki/work/japanese-bert/jawiki-20190701/corpus/corpus.txt.{} --output_file /work/m-suzuki/work/japanese-bert/jawiki-20190701/mecab-ipadic-bpe-32k/max-len-128
/pretraining-data.tf_record.{} --do_whole_word_mask --vocab_file /work/m-suzuki/work/japanese-bert/jawiki-20190701/mecab-ipadic-bpe-32k/vocab.txt --max_seq_length 128 --max_predictions_per_seq 20 --masked_lm_prob 0.15
$ seq -f %02g 0 8|xargs -L 1 -I {} -P 3 python3 create_pretraining_data.py --input_file /work/m-suzuki/work/japanese-bert/jawiki-20190701/corpus/corpus.txt.{} --output_file /work/m-suzuki/work/japanese-bert/jawiki-20190701/mecab-ipadic-bpe-32k/max-len-512/pretraining-data.tf_record.{} --do_whole_word_mask --vocab_file /work/m-suzuki/work/japanese-bert/jawiki-20190701/mecab-ipadic-bpe-32k/vocab.txt --max_seq_length 512 --max_predictions_per_seq 80 --masked_lm_prob 0.15
# 1プロセスあたり30GBのRAMを使用
```

```sh
$ seq -f %02g 0 8|xargs -L 1 -I {} -P 3 python3 create_pretraining_data.py --input_file /work/m-suzuki/work/japanese-bert/jawiki-20190701/corpus/corpus.txt.{} --output_file /work/m-suzuki/work/japanese-bert/jawiki-20190701/mecab-ipadic-char-4k/max-len-128/pretraining-data.tf_record.{} --do_whole_word_mask --vocab_file /work/m-suzuki/work/japanese-bert/jawiki-20190701/mecab-ipadic-char-4k/vocab.txt --subword_type char --max_seq_length 128 --max_predictions_per_seq 20 --masked_lm_prob 0.15
# 1プロセスあたり42GBのRAMを使用
```

### Training

```sh
$ export BASE_DIR='gs://singletongue-tohoku-nlp-2019/japanese-bert/jawiki-20190701/mecab-ipadic-bpe-32k'
$ python3 run_pretraining.py \
--input_file="$BASE_DIR/max-len-128/pretraining-data.tf_record.*" \
--output_dir="$BASE_DIR/max-len-128/outputs" \
--bert_config_file="$BASE_DIR/bert_config.json" \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--do_train=True \
--do_eval=True \
--train_batch_size=256 \
--num_train_steps=1000000 \
--learning_rate=1e-4 \
--save_checkpoints_steps=100000 \
--keep_checkpoint_max=10 \
--use_tpu=True \
--tpu_name=tpu01
$ python3 run_pretraining.py \
--input_file="$BASE_DIR/max-len-512/pretraining-data.tf_record.*" \
--output_dir="$BASE_DIR/max-len-512/outputs/" \
--bert_config_file="$BASE_DIR/bert_config.json" \
--init_checkpoint="$BASE_DIR/max-len-128/outputs/model.ckpt-1000000" \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--do_train=True \
--do_eval=True \
--train_batch_size=256 \
--num_train_steps=100000 \
--num_warmup_steps=10000 \
--learning_rate=1e-4 \
--save_checkpoints_steps=10000 \
--keep_checkpoint_max=10 \
--use_tpu=True \
--tpu_name=tpu01
```

```sh
$ export BASE_DIR='gs://singletongue-tohoku-nlp-2019/japanese-bert/jawiki-20190701/mecab-ipadic-char-4k'
$ python3 run_pretraining.py \
--input_file="$BASE_DIR/max-len-128/pretraining-data.tf_record.*" \
--output_dir="$BASE_DIR/max-len-128/outputs" \
--bert_config_file="$BASE_DIR/bert_config.json" \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--do_train=True \
--do_eval=True \
--train_batch_size=256 \
--num_train_steps=1000000 \
--learning_rate=1e-4 \
--save_checkpoints_steps=100000 \
--keep_checkpoint_max=10 \
--use_tpu=True \
--tpu_name=tpu01
$ python3 run_pretraining.py \
--input_file="$BASE_DIR/max-len-512/pretraining-data.tf_record.*" \
--output_dir="$BASE_DIR/max-len-512/outputs/" \
--bert_config_file="$BASE_DIR/bert_config.json" \
--init_checkpoint="$BASE_DIR/max-len-128/outputs/model.ckpt-1000000" \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--do_train=True \
--do_eval=True \
--train_batch_size=256 \
--num_train_steps=100000 \
--num_warmup_steps=10000 \
--learning_rate=1e-4 \
--save_checkpoints_steps=10000 \
--keep_checkpoint_max=10 \
--use_tpu=True \
--tpu_name=tpu01
```

### Model conversion

```sh
$ cd $BASE_DIR
$ pytorch_transformers bert max-len-512/outputs/model.ckpt-100000 bert_config.json pytorch_model.bin
$ cp max-len-512/outputs/model.ckpt-100000.data-00000-of-00001 model.ckpt.data-00000-of-00001
$ cp max-len-512/outputs/model.ckpt-100000.index model.ckpt.index
$ cp max-len-512/outputs/model.ckpt-100000.meta model.ckpt.meta
$ mkdir mecab-ipadic-bpe-32k
$ cp bert_config.json model.ckpt.* pytorch_model.bin vocab.txt mecab-ipadic-bpe-32k
$ tar czf mecab-ipadic-bpe-32k.tar.gz mecab-ipadic-bpe-32k
```
