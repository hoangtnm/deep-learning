# Transformer-based Experiments for NLP

**Note**: Only `RoBERTa` is supported. Other models will be supported in the future.

### 1.1 Training Tokenizer for a specific language

```sh
export model_dir=models/roberta_vi

python train_tokenizer.py \
    --input_dir data/raw_text
    --output_dir $model_dir \
    --vocab_size 52000
```

### 1.2. Training a language model from scratch

> We’ll train a RoBERTa-like model, which is a BERT-like with a couple of changes (check the [documentation](https://huggingface.co/transformers/model_doc/roberta.html) for more details).

As the model is BERT-like, we’ll train it on a task of `Masked language modeling`, i.e. the predict how to fill arbitrary tokens that we randomly mask in the dataset. This is taken care of by the script.

```sh
export train_data_file=data/raw_text/vi_dedup.txt

python language_modelling.py \
    --train_data_file $train_data_file \
    --output_dir models/experiment \
    --model_type roberta \
    --mlm \
    --config_name $model_dir \
    --tokenizer_name $model_dir \
    --do_train \
    --do_eval \
    --num_train_epochs 10
```