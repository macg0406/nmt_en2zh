import tensorflow as tf

import os
import json
import time
import joblib
import random
from tqdm import tqdm
import numpy as np
from transformer_tf import Transformer, CustomSchedule, create_masks
from vocabulary import tokenization

EPOCHS = 30
# BATCH_SIZE = 64
MAX_TOTAL_LENGTH = 25000
num_layers = 4
d_model = 512
dff = 512
num_heads = 8
dropout_rate = 0.1
checkpoint_path = "./checkpoints/train_en2zh"
data_dump_path = "datasets_en2zh.dat"

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def LOG(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S ||| ", time.localtime()), message)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def load_file(filename, result_list):
    with open(filename) as corpus:
        for line in corpus:
            data = json.loads(line)
            result_list.append((data["english"], data["chinese"]))
            

def encode_sentence(lang1, lang2, tokenizer_en, tokenizer_zh,
                    vocab_size_en, vocab_size_zh):
    token_ids_en = tokenizer_en.convert_tokens_to_ids(tokenizer_en.tokenize(lang1))
    token_ids_zh = tokenizer_zh.convert_tokens_to_ids(tokenizer_zh.tokenize(lang2))
    lang1 = [vocab_size_en] + token_ids_en + [vocab_size_en+1]
    lang2 = [vocab_size_zh] + token_ids_zh + [vocab_size_zh+1]
    return lang1, lang2


def batchify(data_id_list, max_total_length):
    batch_datas = []
    data_id_list.sort(key=lambda x: (len(x[1]), len(x[0])))
    current_bs = 0
    max_len_en = 0
    max_len_zh = 0
    current_batch = []
    for data_ids in data_id_list:
        current_bs += 1
        len_en, len_zh = len(data_ids[0]), len(data_ids[1])
        temp_max_len_en = max(len_en, max_len_en)
        temp_max_len_zh = max(len_zh, max_len_zh)
        if (temp_max_len_en + temp_max_len_zh) * current_bs > max_total_length:
            batch_datas.append(current_batch)
            max_len_en = len_en
            max_len_zh = len_zh
            current_bs = 1
            current_batch = []
        else:
            max_len_en = temp_max_len_en
            max_len_zh = temp_max_len_zh
        current_batch.append(data_ids)
    if current_batch:
        batch_datas.append(current_batch)
    return batch_datas


def read_data():
    if os.path.exists(data_dump_path):
        LOG("Load data from %s." % data_dump_path)
        return joblib.load(data_dump_path)
    
    tokenizer_zh = tokenization.FullTokenizer("vocabulary/zh_vocab.txt")
    tokenizer_en = tokenization.FullTokenizer("vocabulary/en_vocab.txt")
    vocab_size_en = len(tokenizer_en.vocab)
    vocab_size_zh = len(tokenizer_zh.vocab)
    
    train_list = []
    valid_list = []
    load_file("translation2019zh_valid.json", valid_list)
    LOG(len(valid_list))
    # load_file("translation2019zh_valid.json", train_list)
    load_file("translation2019zh_train.json", train_list)
    LOG(len(train_list))
    
    valid_id_list = [encode_sentence(en, zh, tokenizer_en, tokenizer_zh,
                    vocab_size_en, vocab_size_zh) for en, zh in valid_list]
    LOG(len(valid_id_list))
    train_id_list = [encode_sentence(en, zh, tokenizer_en, tokenizer_zh,
                    vocab_size_en, vocab_size_zh) for en, zh in tqdm(train_list, desc="processing traindata")]
    LOG(len(train_id_list))
    
    input_vocab_size = vocab_size_en + 2
    target_vocab_size = vocab_size_zh + 2
    joblib.dump((train_id_list, valid_id_list, input_vocab_size,
                 target_vocab_size), "datasets_en2zh.dat")
    return train_id_list, valid_id_list, input_vocab_size, target_vocab_size


def batch_to_tensor(batch_data):
    batch_size = len(batch_data)
    random.shuffle(batch_data)
    maxlen_0 = max(len(x[0]) for x in batch_data)
    maxlen_1 = max(len(x[1]) for x in batch_data)
    inp = np.zeros((batch_size, maxlen_0), dtype=np.int32)
    tar = np.zeros((batch_size, maxlen_1), dtype=np.int32)
    for i, (en, zh) in enumerate(batch_data):
        inp[i, :len(en)] = en
        tar[i, :len(zh)] = zh
    return tf.convert_to_tensor(inp, dtype=tf.int32), tf.convert_to_tensor(tar, dtype=tf.int32)


def get_tensor_batch(data_list):
    random.shuffle(data_list)
    for batch_data in data_list:
        yield batch_to_tensor(batch_data)


def load_embeddings(path):
    embed_data = joblib.load(path)
    vocab_size, d_model = embed_data.shape
    embedings = np.random.random((vocab_size+2, d_model))
    # embedings[:vocab_size, :] = embed_data
    return embedings


def main():
    train_id_list, valid_id_list, input_vocab_size, target_vocab_size = read_data()
    LOG("Load data finished, %d training, %d validation" %
        (len(train_id_list), len(valid_id_list)))
    train_dataset = batchify(train_id_list, MAX_TOTAL_LENGTH)
    val_dataset = batchify(valid_id_list, MAX_TOTAL_LENGTH)
    LOG(" %d batches of training data, %d batches of validation data." %
        (len(train_dataset), len(val_dataset)))
    # en_embed_data = load_embeddings("vocabulary/en_embedding.dat")
    # zh_embed_data = load_embeddings("vocabulary/zh_embedding.dat")

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.97, beta_2=0.9999)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              # en_embed_data, zh_embed_data,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # train_step_signature = [
    #     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    # ]
    # @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)
            gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    def evaluate_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inp, tar_inp)

        predictions, _ = transformer(inp, tar_inp,
                                     False,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

        train_loss(loss)
        train_accuracy(tar_real, predictions)
    LOG("Params batch size:%d, d_model:%d, dff:%d, num_layers:%d, num_heads:%d." %
        (MAX_TOTAL_LENGTH, d_model, dff, num_layers, num_heads))
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for batch, (inp, tar) in enumerate(get_tensor_batch(train_dataset)):
            train_step(inp, tar)

            if batch % 1000 == 0:
                LOG('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        LOG('Train Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                  train_loss.result(),
                                                                  train_accuracy.result()))

        train_loss.reset_states()
        train_accuracy.reset_states()
        for batch_data in val_dataset:
            inp, tar = batch_to_tensor(batch_data)
            evaluate_step(inp, tar)

        LOG('Valid Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                  train_loss.result(),
                                                                  train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == "__main__":
    main()
