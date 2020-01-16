# nmt_en2zh
Neural Machine Translation (English to Chinese) 
英中机器翻译

## 语料
520万个中英文平行语料，来自于[nlp_chinese_corpus项目](https://github.com/brightmart/nlp_chinese_corpus)

## 模型
基于tensorflow官网上的transformer模型教程：[理解语言的 Transformer 模型](https://www.tensorflow.org/tutorials/text/transformer)，
在此基础上有稍作修改。

## 依赖
- tensorflow>=2.0.0
- joblib
- tqdm

## 训练
将训练语料解压到当前目录，运行
```
python train_transformer_en2zh.py
```
