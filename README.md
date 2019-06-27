```
sudo apt update
sudo apt upgrade
sudo apt install python3-pip parallel mecab libmecab-dev mecab-ipadic-utf8 swig
pip3 install mecab-python3 sentencepiece tensorflow

git clone --recursive https://github.com/singletongue/bert-ja.git
singletongue
fErz89iYMpsN93jy

cd bert-ja

python3 make_wiki_corpus.py gs://singletongue-tohoku-nlp/wikipedia-dump/jawiki-20190225-cirrussearch-content.json.gz gs://singletongue-tohoku-nlp/bert-ja/corpus --mecab_dict=/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd


python3 build_vocab.py "work/corpus/corpus_*.txt" work/bpe-32k/vocab.txt --vocab_type=bpe --vocab_size=32000 --do_lower_case=False

python3 build_vocab.py "work/corpus/corpus_*.txt" work/char/vocab.txt --vocab_type=char --do_lower_case=False

python3 build_vocab.py gs://singletongue-tohoku-nlp/bert-ja/corpus/corpus_*.txt gs://singletongue-tohoku-nlp/bert-ja/bpe-neologd-32k/vocab.txt --vocab_type=bpe --vocab_size=32000 --do_lower_case=False --mecab_dict=/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd


seq -f %03g 0 10|parallel "python3 create_pretraining_data.py --input_file=gs://singletongue-tohoku-nlp/bert-ja/corpus/corpus_{}.txt --output_file=gs://singletongue-tohoku-nlp/bert-ja/bpe-32k/max-len-128/data_{}.tf_record --vocab_file=gs://singletongue-tohoku-nlp/bert-ja/bpe-32k/vocab.txt --do_lower_case=False --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15"

seq -f %03g 0 10|parallel "python3 create_pretraining_data.py --input_file=gs://singletongue-tohoku-nlp/bert-ja/corpus/corpus_{}.txt --output_file=gs://singletongue-tohoku-nlp/bert-ja/bpe-32k/max-len-512/data_{}.tf_record --vocab_file=gs://singletongue-tohoku-nlp/bert-ja/bpe-32k/vocab.txt --do_lower_case=False --max_seq_length=512 --max_predictions_per_seq=80 --masked_lm_prob=0.15"

seq -f %03g 0 10|parallel -j 6 "python3 create_pretraining_data.py --input_file=gs://singletongue-tohoku-nlp/bert-ja/corpus/corpus_{}.txt --output_file=gs://singletongue-tohoku-nlp/bert-ja/bpe-neologd-32k/max-len-512/data_{}.tf_record --vocab_file=gs://singletongue-tohoku-nlp/bert-ja/bpe-neologd-32k/vocab.txt --do_lower_case=False --mecab_dict=/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd --max_seq_length=512 --max_predictions_per_seq=80 --masked_lm_prob=0.15"

seq -f %03g 0 10|parallel -j 6 "python3 create_pretraining_data.py --input_file=gs://singletongue-tohoku-nlp/bert-ja/corpus/corpus_{}.txt --output_file=gs://singletongue-tohoku-nlp/bert-ja/char/max-len-128/data_{}.tf_record --vocab_file=gs://singletongue-tohoku-nlp/bert-ja/char/vocab.txt --vocab_type=char --do_lower_case=False --mecab_dict=/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15"

seq -f %03g 0 10|parallel -j 6 "python3 create_pretraining_data.py --input_file=gs://singletongue-tohoku-nlp/bert-ja/corpus/corpus_{}.txt --output_file=gs://singletongue-tohoku-nlp/bert-ja/char/max-len-512/data_{}.tf_record --vocab_file=gs://singletongue-tohoku-nlp/bert-ja/char/vocab.txt --vocab_type=char --do_lower_case=False --mecab_dict=/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd --max_seq_length=512 --max_predictions_per_seq=80 --masked_lm_prob=0.15"

seq -f %03g 0 10|parallel -j 6 "python3 create_pretraining_data.py --input_file=gs://singletongue-tohoku-nlp/bert-ja/corpus/corpus_{}.txt --output_file=gs://singletongue-tohoku-nlp/bert-ja/char/max-len-512-word-mask/data_{}.tf_record --vocab_file=gs://singletongue-tohoku-nlp/bert-ja/char/vocab.txt --vocab_type=char --do_lower_case=False --mecab_dict=/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd --max_seq_length=512 --max_predictions_per_seq=80 --masked_lm_prob=0.15 --do_mask_words=True"

seq -f %03g 0 10|parallel -j 6 "python3 create_pretraining_data.py --input_file=gs://singletongue-tohoku-nlp/bert-ja/corpus/corpus_{}.txt --output_file=gs://singletongue-tohoku-nlp/bert-ja/char/max-len-512-word-mask-ipadic/data_{}.tf_record --vocab_file=gs://singletongue-tohoku-nlp/bert-ja/char/vocab.txt --vocab_type=char --do_lower_case=False --max_seq_length=512 --max_predictions_per_seq=80 --masked_lm_prob=0.15 --do_mask_words=True"


gsutil -m cp -r gs://singletongue-tohoku-nlp/bert-ja/bpe-32k/max-len-128 gs://singletongue-tohoku-nlp-b1/bert-ja/bpe-32k/
python bert/run_pretraining.py --input_file="gs://singletongue-tohoku-nlp-b1/bert-ja/bpe-32k/max-len-128/data_*.tf_record" --output_dir=gs://singletongue-tohoku-nlp-b1/bert-ja/bpe-32k/max-len-128/output --do_train=True --do_eval=True --bert_config_file=bert_config_base_32k.json --train_batch_size=256 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=1000000 --learning_rate=1e-4 --save_checkpoints_steps=10000 --use_tpu=True --tpu_name=tpu1
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  global_step = 1000000
INFO:tensorflow:  loss = 1.2630663
INFO:tensorflow:  masked_lm_accuracy = 0.73422116
INFO:tensorflow:  masked_lm_loss = 1.2153858
INFO:tensorflow:  next_sentence_accuracy = 0.99875
INFO:tensorflow:  next_sentence_loss = 0.010885029
gsutil -m cp -r gs://singletongue-tohoku-nlp-b1/bert-ja/bpe-32k/max-len-128/output gs://singletongue-tohoku-nlp/bert-ja/bpe-32k/max-len-128
gsutil -m cp -r gs://singletongue-tohoku-nlp/bert-ja/bpe-32k/max-len-512 gs://singletongue-tohoku-nlp-b1/bert-ja/bpe-32k/
python bert/run_pretraining.py --input_file="gs://singletongue-tohoku-nlp-b1/bert-ja/bpe-32k/max-len-512/data_*.tf_record" --output_dir=gs://singletongue-tohoku-nlp-b1/bert-ja/bpe-32k/max-len-512/output --do_train=True --do_eval=True --bert_config_file=bert_config_base_32k.json --train_batch_size=256 --max_seq_length=512 --max_predictions_per_seq=80 --num_train_steps=100000 --learning_rate=1e-4 --save_checkpoints_steps=10000 --use_tpu=True --tpu_name=tpu1 --init_checkpoint=gs://singletongue-tohoku-nlp-b1/bert-ja/bpe-32k/max-len-128/output/model.ckpt-1000000
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  global_step = 100000
INFO:tensorflow:  loss = 1.1959007
INFO:tensorflow:  masked_lm_accuracy = 0.7502446
INFO:tensorflow:  masked_lm_loss = 1.1334296
INFO:tensorflow:  next_sentence_accuracy = 0.99125
INFO:tensorflow:  next_sentence_loss = 0.02344454
gsutil -m cp -r gs://singletongue-tohoku-nlp-b1/bert-ja/bpe-32k/max-len-512/output gs://singletongue-tohoku-nlp/bert-ja/bpe-32k/max-len-128-512/output

gsutil -m cp -r gs://singletongue-tohoku-nlp/bert-ja/char/max-len-512 gs://singletongue-tohoku-nlp-b2/bert-ja/char/
python bert/run_pretraining.py --input_file="gs://singletongue-tohoku-nlp-b2/bert-ja/char/max-len-512/data_*.tf_record" --output_dir=gs://singletongue-tohoku-nlp-b2/bert-ja/char/max-len-512/output --do_train=True --do_eval=True --bert_config_file=bert_config_base_char.json --train_batch_size=256 --max_seq_length=512 --max_predictions_per_seq=80 --num_train_steps=200000 --learning_rate=1e-4 --save_checkpoints_steps=10000 --use_tpu=True --tpu_name=tpu2
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  global_step = 200000
INFO:tensorflow:  loss = 0.74225485
INFO:tensorflow:  masked_lm_accuracy = 0.8275703
INFO:tensorflow:  masked_lm_loss = 0.70348394
INFO:tensorflow:  next_sentence_accuracy = 0.99
INFO:tensorflow:  next_sentence_loss = 0.026093941
gsutil -m cp -r gs://singletongue-tohoku-nlp-b2/bert-ja/char/max-len-512/output gs://singletongue-tohoku-nlp/bert-ja/char/max-len-512
python bert/run_pretraining.py --input_file="gs://singletongue-tohoku-nlp-b2/bert-ja/char/max-len-512/data_*.tf_record" --output_dir=gs://singletongue-tohoku-nlp-b2/bert-ja/char/max-len-512/output --do_train=True --do_eval=True --bert_config_file=bert_config_base_char.json --train_batch_size=256 --max_seq_length=512 --max_predictions_per_seq=80 --num_train_steps=300000 --learning_rate=1e-4 --save_checkpoints_steps=10000 --use_tpu=True --tpu_name=tpu2
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  global_step = 300000
INFO:tensorflow:  loss = 0.68127716
INFO:tensorflow:  masked_lm_accuracy = 0.83663344
INFO:tensorflow:  masked_lm_loss = 0.6577373
INFO:tensorflow:  next_sentence_accuracy = 0.99875
INFO:tensorflow:  next_sentence_loss = 0.010905556
gsutil -m cp -r gs://singletongue-tohoku-nlp-b2/bert-ja/char/max-len-512/output gs://singletongue-tohoku-nlp/bert-ja/char/max-len-512


gsutil -m cp -r gs://singletongue-tohoku-nlp/bert-ja/bpe-32k/max-len-512 gs://singletongue-tohoku-nlp-b3/bert-ja/bpe-32k/
python bert/run_pretraining.py --input_file="gs://singletongue-tohoku-nlp-b3/bert-ja/bpe-32k/max-len-512/data_*.tf_record" --output_dir=gs://singletongue-tohoku-nlp-b3/bert-ja/bpe-32k/max-len-512/output --do_train=True --do_eval=True --bert_config_file=bert_config_base_32k.json --train_batch_size=256 --max_seq_length=512 --max_predictions_per_seq=80 --num_train_steps=200000 --learning_rate=1e-4 --save_checkpoints_steps=10000 --use_tpu=True --tpu_name=tpu3
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  global_step = 200000
INFO:tensorflow:  loss = 1.4193339
INFO:tensorflow:  masked_lm_accuracy = 0.7146252
INFO:tensorflow:  masked_lm_loss = 1.3418813
INFO:tensorflow:  next_sentence_accuracy = 0.98875
INFO:tensorflow:  next_sentence_loss = 0.032133516
gsutil -m cp -r gs://singletongue-tohoku-nlp-b3/bert-ja/bpe-32k/max-len-512 gs://singletongue-tohoku-nlp/bert-ja/bpe-32k/max-len-512
python bert/run_pretraining.py --input_file="gs://singletongue-tohoku-nlp-b3/bert-ja/bpe-32k/max-len-512/data_*.tf_record" --output_dir=gs://singletongue-tohoku-nlp-b3/bert-ja/bpe-32k/max-len-512/output --do_train=True --do_eval=True --bert_config_file=bert_config_base_32k.json --train_batch_size=256 --max_seq_length=512 --max_predictions_per_seq=80 --num_train_steps=300000 --learning_rate=1e-4 --save_checkpoints_steps=10000 --use_tpu=True --tpu_name=tpu3
INFO:tensorflow:  loss = 1.3004764
INFO:tensorflow:  masked_lm_accuracy = 0.73069936
INFO:tensorflow:  masked_lm_loss = 1.2371464
INFO:tensorflow:  next_sentence_accuracy = 0.99625
INFO:tensorflow:  next_sentence_loss = 0.012381399
gsutil -m cp -r gs://singletongue-tohoku-nlp-b3/bert-ja/bpe-32k/max-len-512/output gs://singletongue-tohoku-nlp/bert-ja/bpe-32k/max-len-512

gsutil -m cp -r gs://singletongue-tohoku-nlp/bert-ja/char/max-len-512-word-mask gs://singletongue-tohoku-nlp-b4/bert-ja/char/
python bert/run_pretraining.py --input_file="gs://singletongue-tohoku-nlp-b4/bert-ja/char/max-len-512-word-mask/data_*.tf_record" --output_dir=gs://singletongue-tohoku-nlp-b4/bert-ja/char/max-len-512-word-mask/output --do_train=True --do_eval=True --bert_config_file=bert_config_base_char.json --train_batch_size=256 --max_seq_length=512 --max_predictions_per_seq=80 --num_train_steps=200000 --learning_rate=1e-4 --save_checkpoints_steps=10000 --use_tpu=True --tpu_name=tpu4
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  global_step = 200000
INFO:tensorflow:  loss = 1.8685613
INFO:tensorflow:  masked_lm_accuracy = 0.6171954
INFO:tensorflow:  masked_lm_loss = 1.7655338
INFO:tensorflow:  next_sentence_accuracy = 0.97625
INFO:tensorflow:  next_sentence_loss = 0.052986287
gsutil -m cp -r gs://singletongue-tohoku-nlp-b4/bert-ja/char/max-len-512-word-mask/output gs://singletongue-tohoku-nlp/bert-ja/char/max-len-512-word-mask
python bert/run_pretraining.py --input_file="gs://singletongue-tohoku-nlp-b4/bert-ja/char/max-len-512-word-mask/data_*.tf_record" --output_dir=gs://singletongue-tohoku-nlp-b4/bert-ja/char/max-len-512-word-mask/output --do_train=True --do_eval=True --bert_config_file=bert_config_base_char.json --train_batch_size=256 --max_seq_length=512 --max_predictions_per_seq=80 --num_train_steps=300000 --learning_rate=1e-4 --save_checkpoints_steps=10000 --use_tpu=True --tpu_name=tpu4
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  global_step = 300000
INFO:tensorflow:  loss = 1.7845131
INFO:tensorflow:  masked_lm_accuracy = 0.6298905
INFO:tensorflow:  masked_lm_loss = 1.6963465
INFO:tensorflow:  next_sentence_accuracy = 0.9875
INFO:tensorflow:  next_sentence_loss = 0.034039002
gsutil -m cp -r gs://singletongue-tohoku-nlp-b4/bert-ja/char/max-len-512-word-mask/output gs://singletongue-tohoku-nlp/bert-ja/char/max-len-512-word-mask


gsutil -m cp -r gs://singletongue-tohoku-nlp/bert-ja/bpe-neologd-32k/max-len-128 gs://singletongue-tohoku-nlp-b1/bert-ja/bpe-neologd-32k/
python bert/run_pretraining.py --input_file="gs://singletongue-tohoku-nlp-b1/bert-ja/bpe-neologd-32k/max-len-128/data_*.tf_record" --output_dir=gs://singletongue-tohoku-nlp-b1/bert-ja/bpe-neologd-32k/max-len-128/output --do_train=True --do_eval=True --bert_config_file=bert_config_base_32k.json --train_batch_size=256 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=1000000 --learning_rate=1e-4 --save_checkpoints_steps=10000 --use_tpu=True --tpu_name=tpu1


tpu1 -> bpe-neologd-32k/max-len-128 [1000000]
tpu2 -> char/max-len-512 [300000]
tpu3 -> bpe-32k/max-len-512 [300000]
tpu4 -> char/max-len-512-word-mask [300000]
```
