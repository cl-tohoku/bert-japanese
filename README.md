# Pretrained Japanese BERT models

This is a repository of pretrained Japanese BERT models.
The models are available in [Transformers](https://github.com/huggingface/transformers) by Hugging Face.

- Model hub: https://huggingface.co/cl-tohoku

This version of README contains information for the following models:

- [`cl-tohoku/bert-base-japanese-v3`](https://huggingface.co/cl-tohoku/bert-base-japanese-v3)
- [`cl-tohoku/bert-base-japanese-char-v3`](https://huggingface.co/cl-tohoku/bert-base-japanese-char-v3)
- [`cl-tohoku/bert-large-japanese-v2`](https://huggingface.co/cl-tohoku/bert-large-japanese-v2)
- [`cl-tohoku/bert-large-japanese-char-v2`](https://huggingface.co/cl-tohoku/bert-large-japanese-char-v2)

For information and codes for the following models, refer to the [v2.0](https://github.com/cl-tohoku/bert-japanese/tree/v2.0) tag of this repository:

- [`cl-tohoku/bert-base-japanese-v2`](https://huggingface.co/cl-tohoku/bert-base-japanese-v2)
- [`cl-tohoku/bert-base-japanese-char-v2`](https://huggingface.co/cl-tohoku/bert-base-japanese-char-v2)
- [`cl-tohoku/bert-large-japanese`](https://huggingface.co/cl-tohoku/bert-large-japanese)
- [`cl-tohoku/bert-large-japanese-char`](https://huggingface.co/cl-tohoku/bert-large-japanese-char)

For information and codes for the following models, refer to the [v1.0](https://github.com/cl-tohoku/bert-japanese/tree/v1.0) tag of this repository:

- [`cl-tohoku/bert-base-japanese`](https://huggingface.co/cl-tohoku/bert-base-japanese)
- [`cl-tohoku/bert-base-japanese-whole-word-masking`](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking)
- [`cl-tohoku/bert-base-japanese-char`](https://huggingface.co/cl-tohoku/bert-base-japanese-char)
- [`cl-tohoku/bert-base-japanese-char-whole-word-masking`](https://huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking)

## Model Architecture

The architecture of our models are the same as the original BERT models proposed by Google.
- **BERT-base** models consist of 12 layers, 768 dimensions of hidden states, and 12 attention heads.
- **BERT-large** models consist of 24 layers, 1024 dimensions of hidden states, and 16 attention heads.

## Training Data

The models are trained on the Japanese portion of [CC-100 dataset](https://data.statmt.org/cc-100/) and the Japanese version of Wikipedia.
For Wikipedia, we generated a text corpus from the [Wikipedia Cirrussearch dump file](https://dumps.wikimedia.org/other/cirrussearch/) as of January 2, 2023.

The corpus files generated from CC-100 and Wikipedia are 74.3GB and 4.9GB in size and consist of approximately 392M and 34M sentences, respectively.

For the purpose of splitting texts into sentences, we used [`fugashi`](https://github.com/polm/fugashi) with [mecab-ipadic-NEologd](https://github.com/neologd/mecab-ipadic-neologd) dictionary (v0.0.7).

### Generating corpus files

```sh
# For CC-100
$ mkdir -p $WORK_DIR/corpus/cc-100
$ python merge_split_corpora.py \
--input_files $DATA_DIR/cc-100/ja.txt.xz \
--output_dir $WORK_DIR/corpus/cc-100 \
--num_files 64

# For Wikipedia
$ mkdir -p $WORK_DIR/corpus/wikipedia
$ python make_corpus_wiki.py \
--input_file $DATA_DIR/wikipedia/cirrussearch/20230102/jawiki-20230102-cirrussearch-content.json.gz \
--output_file $WORK_DIR/corpus/wikipedia/corpus.txt.gz \
--min_sentence_length 10 \
--max_sentence_length 200 \
--mecab_option '-r <path to etc/mecabrc> -d <path to lib/mecab/dic/mecab-ipadic-neologd>'
$ python merge_split_corpora.py \
--input_files $WORK_DIR/corpus/wikipedia/corpus.txt.gz \
--output_dir $WORK_DIR/corpus/wikipedia \
--num_files 8

# Sample 1M sentences for training tokenizers
$ cat $WORK_DIR/corpus/wikipedia/corpus_*.txt|grep -a -v '^$'|shuf|head -n 10000000 > $WORK_DIR/corpus/wikipedia/corpus_sampled.txt
```

## Tokenization

For each of BERT-base and BERT-large, we provide two models with different tokenization methods.

- For **`wordpiece`** models, the texts are first tokenized by MeCab with the Unidic 2.1.2 dictionary and then split into subwords by the WordPiece algorithm.
  The vocabulary size is 32768.
- For **`character`** models, the texts are first tokenized by MeCab with the Unidic 2.1.2 dictionary and then split into characters.
  The vocabulary size is 7027, which covers all the characters present in Unidic 2.1.2 dictionary.

We used [`unidic-lite`](https://github.com/polm/unidic-lite) dictionary for tokenization.

### Generating a set of characters

```sh
$ mkdir -p $WORK_DIR/tokenizers/alphabet
$ python make_alphabet_from_unidic.py \
--lex_file $DATA_DIR/unidic-mecab-2.1.2_src/lex.csv \
--output_file $WORK_DIR/tokenizers/alphabet/unidic_lite.txt
```

### Training tokenizers

```sh
# WordPiece
$ python train_tokenizer.py \
--input_files $WORK_DIR/corpus/wikipedia/corpus_sampled.txt \
--output_dir $WORK_DIR/tokenizers/wordpiece_unidic_lite \
--pre_tokenizer_type mecab \
--mecab_dic_type unidic_lite \
--vocab_size 32768 \
--limit_alphabet 7012 \
--initial_alphabet_file $WORK_DIR/tokenizers/alphabet/unidic_lite.txt \
--num_unused_tokens 10 \
--wordpieces_prefix '##'

# Character
$ mkdir $WORK_DIR/tokenizers/character_unidic_lite
$ head -n 7027 $WORK_DIR/tokenizers/wordpiece_unidic_lite/vocab.txt > $WORK_DIR/tokenizers/character_unidic_lite/vocab.txt
```

### Generating pretraining data

```sh
# WordPiece on CC-100
# Each process takes about 2h50m and 60GB RAM, producing 15.2M instances
$ mkdir -p $WORK_DIR/pretraining_data/wordpiece_unidic_lite/cc-100
$ seq -f %02g 1 64|xargs -L 1 -I {} -P 2 \
python create_pretraining_data.py \
--input_file $WORK_DIR/corpus/cc-100/corpus_{}.txt \
--output_file $WORK_DIR/pretraining_data/wordpiece_unidic_lite/cc-100/pretraining_data_{}.tfrecord.gz \
--vocab_file $WORK_DIR/tokenizers/wordpiece_unidic_lite/vocab.txt \
--word_tokenizer_type mecab \
--subword_tokenizer_type wordpiece \
--mecab_dic_type unidic_lite \
--do_whole_word_mask \
--gzip_compress \
--use_v2_feature_names \
--max_seq_length 128 \
--max_predictions_per_seq 19 \
--masked_lm_prob 0.15 \
--dupe_factor 5

# WordPiece on Wikipedia
# Each process takes about 7h30m and 138GB RAM, producing 18.4M instances
$ mkdir -p $WORK_DIR/pretraining_data/wordpiece_unidic_lite/wikipedia
$ seq -f %02g 1 8|xargs -L 1 -I {} -P 1 \
python create_pretraining_data.py \
--input_file $WORK_DIR/corpus/wikipedia/corpus_{}.txt \
--output_file $WORK_DIR/pretraining_data/wordpiece_unidic_lite/wikipedia/pretraining_data_{}.tfrecord.gz \
--vocab_file $WORK_DIR/tokenizers/wordpiece_unidic_lite/vocab.txt \
--word_tokenizer_type mecab \
--subword_tokenizer_type wordpiece \
--mecab_dic_type unidic_lite \
--do_whole_word_mask \
--gzip_compress \
--use_v2_feature_names \
--max_seq_length 512 \
--max_predictions_per_seq 76 \
--masked_lm_prob 0.15 \
--dupe_factor 30

# Character on CC-100
# Each process takes about 3h30m and 82GB RAM, producing 18.4M instances
$ mkdir -p $WORK_DIR/pretraining_data/character_unidic_lite/cc-100
$ seq -f %02g 1 64|xargs -L 1 -I {} -P 2 \
python create_pretraining_data.py \
--input_file $WORK_DIR/corpus/cc-100/corpus_{}.txt \
--output_file $WORK_DIR/pretraining_data/character_unidic_lite/cc-100/pretraining_data_{}.tfrecord.gz \
--vocab_file $WORK_DIR/tokenizers/character_unidic_lite/vocab.txt \
--word_tokenizer_type mecab \
--subword_tokenizer_type character \
--mecab_dic_type unidic_lite \
--vocab_has_no_subword_prefix \
--do_whole_word_mask \
--gzip_compress \
--use_v2_feature_names \
--max_seq_length 128 \
--max_predictions_per_seq 19 \
--masked_lm_prob 0.15 \
--dupe_factor 5

# Character on Wikipedia
# Each process takes about 10h30m and 205GB RAM, producing 23.7M instances
$ mkdir -p $WORK_DIR/pretraining_data/character_unidic_lite/wikipedia
$ seq -f %02g 1 8|xargs -L 1 -I {} -P 1 \
python create_pretraining_data.py \
--input_file $WORK_DIR/corpus/wikipedia/corpus_{}.txt \
--output_file $WORK_DIR/pretraining_data/character_unidic_lite/wikipedia/pretraining_data_{}.tfrecord.gz \
--vocab_file $WORK_DIR/tokenizers/character_unidic_lite/vocab.txt \
--word_tokenizer_type mecab \
--subword_tokenizer_type character \
--mecab_dic_type unidic_lite \
--vocab_has_no_subword_prefix \
--do_whole_word_mask \
--gzip_compress \
--use_v2_feature_names \
--max_seq_length 512 \
--max_predictions_per_seq 76 \
--masked_lm_prob 0.15 \
--dupe_factor 30
```

## Training

We trained the models first on the CC-100 corpus and then on the Wikipedia corpus.
Generally speaking, the texts of Wikipedia are much cleaner than those of CC-100, but the amount of text is much smaller.
We expect that our two-stage training scheme let the model trained on large amount of text while preserving the quality of language that the model eventually learns.

For training of the MLM (masked language modeling) objective, we introduced **whole word masking** in which all subword tokens corresponding to a single word (tokenized by MeCab) are masked at once.

To conduct training of each model, we used a v3-8 instance of Cloud TPUs provided by [TensorFlow Research Cloud program](https://www.tensorflow.org/tfrc/).
The training took about 16 and 56 days for BERT-base and BERT-large models, respectively.

### Creating a TPU VM and connecting to it

**Note:** We set the runtime version of the TPU as `2.11.0`, where TensorFlow v2.11 is used.
It is important to specify the same version if you wish to reuse our codes, otherwise it may not work properly.

Here we use [Google Cloud CLI](https://cloud.google.com/cli).

```sh
$ gcloud compute tpus tpu-vm create <TPU_NODE_ID> --zone=<TPU_ZONE> --accelerator-type=v3-8 --version=tpu-vm-tf-2.11.0
$ gcloud compute tpus tpu-vm ssh <TPU_NODE_ID> --zone=<TPU_ZONE>
```

### Training of the models

The following commands are executed in the TPU VM.
It is recommended that you run the commands in a Tmux session.

**Note:** All the necessary files (i.e., pretraining data and config files) need to be stored in a Google Cloud Storage (GCS) bucket in advance.

#### BERT-base, WordPiece

```sh
(vm)$ cd /usr/share/tpu/models/
(vm)$ pip3 install -r official/requirements.txt
(vm)$ export PYTHONPATH=/usr/share/tpu/models
(vm)$ CONFIG_DIR="gs://<GCS_BUCKET_ID>/bert-japanese/configs"
(vm)$ DATA_DIR="gs://<GCS_BUCKET_ID>/bert-japanese/pretraining_data/wordpiece_unidic_lite"
(vm)$ MODEL_DIR="gs://<GCS_BUCKET_ID>/bert-japanese/model/wordpiece_unidic_lite"

# Start training on CC-100
# It will take 6 days to finish on a v3-8 TPU
(vm)$ python3 official/nlp/train.py \
--tpu=local \
--experiment=bert/pretraining \
--mode=train_and_eval \
--model_dir=$MODEL_DIR/bert_base/training/cc-100 \
--config_file=$CONFIG_DIR/data/cc-100.yaml \
--config_file=$CONFIG_DIR/model/bert_base_wordpiece.yaml \
--params_override="task.train_data.input_path=$DATA_DIR/cc-100/pretraining_data_*.tfrecord,task.validation_data.input_path=$DATA_DIR/cc-100/pretraining_data_*.tfrecord,runtime.distribution_strategy=tpu"

# Continue training on Wikipedia
# It will take 10 days to finish on a v3-8 TPU
(vm)$ python3 official/nlp/train.py \
--tpu=local \
--experiment=bert/pretraining \
--mode=train_and_eval \
--model_dir=$MODEL_DIR/bert_base/training/cc-100_wikipedia \
--config_file=$CONFIG_DIR/data/wikipedia.yaml \
--config_file=$CONFIG_DIR/model/bert_base_wordpiece.yaml \
--params_override="task.init_checkpoint=$MODEL_DIR/bert_base/training/cc-100,task.train_data.input_path=$DATA_DIR/wikipedia/pretraining_data_*.tfrecord,task.validation_data.input_path=$DATA_DIR/wikipedia/pretraining_data_*.tfrecord,runtime.distribution_strategy=tpu"
```

#### BERT-base, Character

```sh
(vm)$ cd /usr/share/tpu/models/
(vm)$ pip3 install -r official/requirements.txt
(vm)$ export PYTHONPATH=/usr/share/tpu/models
(vm)$ CONFIG_DIR="gs://<GCS_BUCKET_ID>/bert-japanese/configs"
(vm)$ DATA_DIR="gs://<GCS_BUCKET_ID>/bert-japanese/pretraining_data/character_unidic_lite"
(vm)$ MODEL_DIR="gs://<GCS_BUCKET_ID>/bert-japanese/model/character_unidic_lite"

# Start training on CC-100
# It will take 6 days to finish on a v3-8 TPU
(vm)$ python3 official/nlp/train.py \
--tpu=local \
--experiment=bert/pretraining \
--mode=train_and_eval \
--model_dir=$MODEL_DIR/bert_base/training/cc-100 \
--config_file=$CONFIG_DIR/data/cc-100.yaml \
--config_file=$CONFIG_DIR/model/bert_base_character.yaml \
--params_override="task.train_data.input_path=$DATA_DIR/cc-100/pretraining_data_*.tfrecord,task.validation_data.input_path=$DATA_DIR/cc-100/pretraining_data_*.tfrecord,runtime.distribution_strategy=tpu"

# Continue training on Wikipedia
# It will take 10 days to finish on a v3-8 TPU
(vm)$ python3 official/nlp/train.py \
--tpu=local \
--experiment=bert/pretraining \
--mode=train_and_eval \
--model_dir=$MODEL_DIR/bert_base/training/cc-100_wikipedia \
--config_file=$CONFIG_DIR/data/wikipedia.yaml \
--config_file=$CONFIG_DIR/model/bert_base_character.yaml \
--params_override="task.init_checkpoint=$MODEL_DIR/bert_base/training/cc-100,task.train_data.input_path=$DATA_DIR/wikipedia/pretraining_data_*.tfrecord,task.validation_data.input_path=$DATA_DIR/wikipedia/pretraining_data_*.tfrecord,runtime.distribution_strategy=tpu"
```

#### BERT-large, WordPiece

```sh
(vm)$ cd /usr/share/tpu/models/
(vm)$ pip3 install -r official/requirements.txt
(vm)$ export PYTHONPATH=/usr/share/tpu/models
(vm)$ CONFIG_DIR="gs://<GCS_BUCKET_ID>/bert-japanese/configs"
(vm)$ DATA_DIR="gs://<GCS_BUCKET_ID>/bert-japanese/pretraining_data/wordpiece_unidic_lite"
(vm)$ MODEL_DIR="gs://<GCS_BUCKET_ID>/bert-japanese/model/wordpiece_unidic_lite"

# Start training on CC-100
# It will take 23 days to finish on a v3-8 TPU
(vm)$ python3 official/nlp/train.py \
--tpu=local \
--experiment=bert/pretraining \
--mode=train_and_eval \
--model_dir=$MODEL_DIR/bert_large/training/cc-100 \
--config_file=$CONFIG_DIR/data/cc-100.yaml \
--config_file=$CONFIG_DIR/model/bert_large_wordpiece.yaml \
--params_override="task.train_data.input_path=$DATA_DIR/cc-100/pretraining_data_*.tfrecord,task.validation_data.input_path=$DATA_DIR/cc-100/pretraining_data_*.tfrecord,runtime.distribution_strategy=tpu"

# Continue training on Wikipedia
# It will take 33 days to finish on a v3-8 TPU
(vm)$ python3 official/nlp/train.py \
--tpu=local \
--experiment=bert/pretraining \
--mode=train_and_eval \
--model_dir=$MODEL_DIR/bert_large/training/cc-100_wikipedia \
--config_file=$CONFIG_DIR/data/wikipedia.yaml \
--config_file=$CONFIG_DIR/model/bert_large_wordpiece.yaml \
--params_override="task.init_checkpoint=$MODEL_DIR/bert_large/training/cc-100,task.train_data.input_path=$DATA_DIR/wikipedia/pretraining_data_*.tfrecord,task.validation_data.input_path=$DATA_DIR/wikipedia/pretraining_data_*.tfrecord,runtime.distribution_strategy=tpu"
```

#### BERT-large, Character

```sh
(vm)$ cd /usr/share/tpu/models/
(vm)$ pip3 install -r official/requirements.txt
(vm)$ export PYTHONPATH=/usr/share/tpu/models
(vm)$ CONFIG_DIR="gs://<GCS_BUCKET_ID>/bert-japanese/configs"
(vm)$ DATA_DIR="gs://<GCS_BUCKET_ID>/bert-japanese/pretraining_data/character_unidic_lite"
(vm)$ MODEL_DIR="gs://<GCS_BUCKET_ID>/bert-japanese/model/character_unidic_lite"

# Start training on CC-100
# It will take 23 days to finish on a v3-8 TPU
(vm)$ python3 official/nlp/train.py \
--tpu=local \
--experiment=bert/pretraining \
--mode=train_and_eval \
--model_dir=$MODEL_DIR/bert_large/training/cc-100 \
--config_file=$CONFIG_DIR/data/cc-100.yaml \
--config_file=$CONFIG_DIR/model/bert_large_character.yaml \
--params_override="task.train_data.input_path=$DATA_DIR/cc-100/pretraining_data_*.tfrecord,task.validation_data.input_path=$DATA_DIR/cc-100/pretraining_data_*.tfrecord,runtime.distribution_strategy=tpu"

# Continue training on Wikipedia
# It will take 33 days to finish on a v3-8 TPU
(vm)$ python3 official/nlp/train.py \
--tpu=local \
--experiment=bert/pretraining \
--mode=train_and_eval \
--model_dir=$MODEL_DIR/bert_large/training/cc-100_wikipedia \
--config_file=$CONFIG_DIR/data/wikipedia.yaml \
--config_file=$CONFIG_DIR/model/bert_large_character.yaml \
--params_override="task.init_checkpoint=$MODEL_DIR/bert_large/training/cc-100,task.train_data.input_path=$DATA_DIR/wikipedia/pretraining_data_*.tfrecord,task.validation_data.input_path=$DATA_DIR/wikipedia/pretraining_data_*.tfrecord,runtime.distribution_strategy=tpu"
```

### Deleting a TPU VM

```sh
$ gcloud compute tpus tpu-vm delete <TPU_NODE_ID> --zone=<TPU_ZONE>
```

## Model Conversion

You can convert the TensorFlow model checkpoint to a PyTorch model file.

**Note:** The model conversion script is designed for the models trained with TensorFlow v2.11.0.
The script may not work for models trained with a different version of TensorFlow.

```sh
# For BERT-base, WordPiece
$ VOCAB_FILE=$WORK_DIR/tokenizers/wordpiece_unidic_lite/vocab.txt
$ TF_CONFIG_FILE=model_configs/bert_base_wordpiece/config.json
$ HF_CONFIG_DIR=hf_model_configs/bert_base_wordpiece
$ TF_CKPT_PATH=$WORK_DIR/model/wordpiece_unidic_lite/bert_base/training/cc-100_wikipedia/ckpt-1000000
$ OUTPUT_DIR=$WORK_DIR/hf_model/wordpiece_unidic_lite/bert_base

# For BERT-base, Character
$ VOCAB_FILE=$WORK_DIR/tokenizers/character_unidic_lite/vocab.txt
$ TF_CONFIG_FILE=model_configs/bert_base_character/config.json
$ HF_CONFIG_DIR=hf_model_configs/bert_base_character
$ TF_CKPT_PATH=$WORK_DIR/model/character_unidic_lite/bert_base/training/cc-100_wikipedia/ckpt-1000000
$ OUTPUT_DIR=$WORK_DIR/hf_model/character_unidic_lite/bert_base

# For BERT-large, WordPiece
$ VOCAB_FILE=$WORK_DIR/tokenizers/wordpiece_unidic_lite/vocab.txt
$ TF_CONFIG_FILE=model_configs/bert_large_wordpiece/config.json
$ HF_CONFIG_DIR=hf_model_configs/bert_large_wordpiece
$ TF_CKPT_PATH=$WORK_DIR/model/wordpiece_unidic_lite/bert_large/training/cc-100_wikipedia/ckpt-1000000
$ OUTPUT_DIR=$WORK_DIR/hf_model/wordpiece_unidic_lite/bert_large

# For BERT-large, Character
$ VOCAB_FILE=$WORK_DIR/tokenizers/character_unidic_lite/vocab.txt
$ TF_CONFIG_FILE=model_configs/bert_large_character/config.json
$ HF_CONFIG_DIR=hf_model_configs/bert_large_character
$ TF_CKPT_PATH=$WORK_DIR/model/character_unidic_lite/bert_large/training/cc-100_wikipedia/ckpt-1000000
$ OUTPUT_DIR=$WORK_DIR/hf_model/character_unidic_lite/bert_large

# Run the model conversion script
$ mkdir -p $OUTPUT_DIR
$ cp $HF_CONFIG_DIR/* $OUTPUT_DIR
$ cp $VOCAB_FILE $OUTPUT_DIR
$ python convert_original_tf2_checkpoint_to_pytorch.py \
--tf_checkpoint_path $TF_CKPT_PATH \
--pytorch_dump_path $OUTPUT_DIR/pytorch_model.bin \
--config_file $TF_CONFIG_FILE
```

## Model Performances

We evaluated the models' performances on the [JGLUE](https://github.com/yahoojapan/JGLUE) benchmark tasks.

For each task, the model is fine-tuned on the training set and evaluated on the development set (the test sets are not publicly available as of this writing.)
We used the same hyperparameters as the ones specified in the [JGLUE fine-tuning README](https://github.com/yahoojapan/JGLUE/tree/main/fine-tuning).

The results of our (informal) experiments are below.
Since each setting is experimented with only once (random seed is fixed), these results should be viewed only as a reference.

|              Model                     | MARC-ja |        JSTS        | JNLI  |    JSQuAD     | JCommonsenseQA |
| :------------------------------------- | :-----: | :----------------: | :---: | :-----------: | :------------: |
|                                        |   Acc.  | Pearson / Spearman |  Acc. |    EM / F1    |       Acc.     |
| `bert-base-japanese-v2`                |  0.952  |   0.907 / 0.867    | 0.897 | 0.873 / 0.941 |      0.802     |
| `bert-base-japanese-char-v2`           |  0.954  |   0.872 / 0.893    | 0.892 | 0.862 / 0.936 |      0.720     |
| `bert-large-japanese`                  |  0.955  |   0.910 / 0.871    | 0.901 | 0.873 / 0.943 |      0.803     |
| `bert-large-japanese-char`             |  0.956  |   0.874 / 0.834    | 0.899 | 0.871 / 0.940 |      0.741     |
|                                        |         |                    |       |               |                |
| `bert-base-japanese-v3` **New!**       |  0.960  |   0.915 / 0.878    | 0.906 | 0.880 / 0.947 |      0.837     |
| `bert-base-japanese-char-v3` **New!**  |  0.955  |   0.912 / 0.875    | 0.899 | 0.865 / 0.937 |      0.775     |
| `bert-large-japanese-v2` **New!**      |  0.960  |   0.927 / 0.893    | 0.927 | 0.889 / 0.954 |      0.891     |
| `bert-large-japanese-char-v2` **New!** |  0.959  |   0.918 / 0.881    | 0.909 | 0.888 / 0.950 |      0.854     |

## Licenses

The pretrained models and the codes in this repository are distributed under the Apache License 2.0.

## Related Work

- Original BERT models and codes by Google Research Team
    - https://github.com/google-research/bert (for TensorFlow v1)
    - https://github.com/tensorflow/models/tree/master/official/nlp (for TensorFlow v2)

## Acknowledgments

The distributed models are trained with Cloud TPUs provided by [TPU Research Cloud](https://sites.research.google/trc/about/) program.
