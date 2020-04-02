# Named Entity Recognition

The CoNLL-2003 shared task data files contain four columns separated by a single space. Each word has been put on a separate line and there is an empty line after each sentence.

In a pre-processing step only the two relevant columns (token and outer span NER annotation) are extracted:

```sh
export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased

cat train.txt | grep -v "^#" | cut -f 1,4 | tr '\t' ' ' > train.txt.tmp
python utils/ner_preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > train.txt
```
