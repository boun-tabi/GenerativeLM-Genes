import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_nlp

import os
import pickle
import random
import pandas as pd

# The following resources were useful
# - https://keras.io/examples/generative/text_generation_gpt/
# - https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object
# - https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
# - https://stackoverflow.com/questions/18221436/efficient-way-to-add-spaces-between-characters-in-a-string
# - https://stackoverflow.com/questions/18221436/efficient-way-to-add-spaces-between-characters-in-a-string
# - https://stackoverflow.com/questions/61078946/how-to-get-reproducible-results-keras-tensorflow

def main():
    make_reproducible()
    WILL_DIRECTLY_EVALUATE = True
    prepare_actual_datasets()

    BATCH_SIZE, SEQ_LEN, MIN_TRAINING_SEQ_LEN, EMBED_DIM, FEED_FORWARD_DIM, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE, EPOCHS, NUM_TOKENS_TO_GENERATE, LR = initialize_hyper_parameters()

    raw_train_ds, raw_val_ds, raw_test_ds = prepare_dataset(MIN_TRAINING_SEQ_LEN, BATCH_SIZE)
    print('Dataset was prepared')
    vocab = get_vocab(raw_train_ds, VOCAB_SIZE)
    print('Vocabulary was prepared')
    tokenizer = get_tokenizer(vocab, SEQ_LEN)
    print('Tokenizer was prepared')
    train_ds, val_ds, test_ds = tokenize_dataset(SEQ_LEN, tokenizer, raw_train_ds, raw_val_ds, raw_test_ds)
    print('Dataset was tokenized')

    if WILL_DIRECTLY_EVALUATE:
        print('Model is being read')
        model = keras.models.load_model("./my_model")
        print('Model was read')
        do_inference(model, tokenizer, NUM_TOKENS_TO_GENERATE, test_ds, 'Test')
    else:
        model = build_model(VOCAB_SIZE, SEQ_LEN, EMBED_DIM, NUM_LAYERS, NUM_HEADS, FEED_FORWARD_DIM, LR)
        print('Model was built')
        history = train_model(model, train_ds, val_ds, EPOCHS)
        with open('./trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        print('Model was trained')
        model.save("./my_model")
        do_inference(model, tokenizer, NUM_TOKENS_TO_GENERATE, val_ds, 'Validation')
    print('Inference was made')

def initialize_hyper_parameters():
    BATCH_SIZE = 64
    SEQ_LEN = 2**10 - 1
    MIN_TRAINING_SEQ_LEN = 2

    EMBED_DIM = 256
    FEED_FORWARD_DIM = 256
    NUM_HEADS = 3
    NUM_LAYERS = 2
    VOCAB_SIZE = 8
    EPOCHS = 40
    LR = 1e-3

    NUM_TOKENS_TO_GENERATE = SEQ_LEN - 1

    return BATCH_SIZE, SEQ_LEN, MIN_TRAINING_SEQ_LEN, EMBED_DIM, FEED_FORWARD_DIM, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE, EPOCHS, NUM_TOKENS_TO_GENERATE, LR

def make_reproducible():
    seed_value = 364187
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    tf.keras.utils.set_random_seed(seed_value)
    tf.config.experimental.enable_op_determinism()

def prepare_actual_datasets():
    prepare_actual_dataset('./actual_train_without_val_dataset.csv', 'train')
    prepare_actual_dataset('./actual_val_dataset.csv', 'validation')
    prepare_actual_dataset('./actual_test_dataset.csv', 'test')

def prepare_actual_dataset(path, name):
    dataframe = pd.read_csv(path)
    gene_nucleotide_sequences = dataframe['NucleotideSequence']
    gene_nucleotide_sequences_list = gene_nucleotide_sequences.tolist()

    if not os.path.exists('./Datasets'):
        os.mkdir('./Datasets')
    f = open('./Datasets/' + name + '.txt', 'w')
    for i in range(len(gene_nucleotide_sequences_list)):
        f.write(' '.join(gene_nucleotide_sequences_list[i][1:-1]))
        f.write('\n\n')

        if i % 100 == 0:
            print(i, '/', len(gene_nucleotide_sequences_list))

    f.close()

def prepare_dataset(MIN_TRAINING_SEQ_LEN, BATCH_SIZE):
    dir = os.path.expanduser("./Datasets/")

    raw_train_ds = (
        tf.data.TextLineDataset(dir + "train.txt")
            .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
            .batch(BATCH_SIZE)
            .shuffle(buffer_size=256)
    )
    raw_val_ds = (
        tf.data.TextLineDataset(dir + "validation.txt")
            .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
            .batch(BATCH_SIZE)
    )

    raw_test_ds = (
        tf.data.TextLineDataset(dir + "test.txt")
            .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
            .batch(BATCH_SIZE)
    )

    return raw_train_ds, raw_val_ds, raw_test_ds

def get_vocab(raw_train_ds, VOCAB_SIZE):
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        raw_train_ds,
        vocabulary_size=VOCAB_SIZE,
        lowercase=True,
        reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
    )
    return vocab

def get_tokenizer(vocab, SEQ_LEN):
    tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary=vocab,
        sequence_length=SEQ_LEN,
        lowercase=True,
    )
    return tokenizer

def tokenize_dataset(SEQ_LEN, tokenizer, raw_train_ds, raw_val_ds, raw_test_ds):
    start_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=SEQ_LEN,
        start_value=tokenizer.token_to_id("[BOS]"),
    )

    def preprocess(inputs):
        outputs = tokenizer(inputs)
        features = start_packer(outputs)
        labels = outputs
        return features, labels

    train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE
    )
    val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE
    )

    test_ds = raw_test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE
    )

    return train_ds, val_ds, test_ds

def build_model(VOCAB_SIZE, SEQ_LEN, EMBED_DIM, NUM_LAYERS, NUM_HEADS, FEED_FORWARD_DIM, LR):
    inputs = keras.layers.Input(shape=(None,), dtype=tf.int32)
    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=VOCAB_SIZE,
        sequence_length=SEQ_LEN,
        embedding_dim=EMBED_DIM,
        mask_zero=True,
    )
    x = embedding_layer(inputs)
    for _ in range(NUM_LAYERS):
        decoder_layer = keras_nlp.layers.TransformerDecoder(
            num_heads=NUM_HEADS,
            intermediate_dim=FEED_FORWARD_DIM,
        )
        x = decoder_layer(x)
    outputs = keras.layers.Dense(VOCAB_SIZE)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss=loss_fn, metrics=[perplexity])
    return model

def train_model(model, train_ds, val_ds, EPOCHS):
    print(model.summary())
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(train_ds, validation_data=val_ds, verbose=2, epochs=EPOCHS, callbacks=[stop_early])
    return history

def do_inference(model, tokenizer, NUM_TOKENS_TO_GENERATE, ds, name):
    prompt_tokens = tf.convert_to_tensor([tokenizer.token_to_id("[BOS]")])
    res = model.evaluate(ds)
    print(name, 'Set Result:', res)

    def token_logits_fn(inputs):
        cur_len = inputs.shape[1]
        output = model(inputs)
        return output[:, cur_len - 1, :]

    output_tokens = keras_nlp.utils.greedy_search(
        token_logits_fn,
        prompt_tokens,
        max_length=NUM_TOKENS_TO_GENERATE,
    )
    txt = tokenizer.detokenize(output_tokens)
    # print(f"Greedy search generated text: \n{txt}\n")

if __name__ == '__main__':
    main()
