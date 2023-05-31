import numpy as np

import gc
import pickle
import time
import pandas as pd
import os
import random

import nltk
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import ngrams

import tensorflow as tf
from tensorflow import keras
import keras_nlp

# The following resources were useful
# - https://keras.io/examples/generative/text_generation_gpt/
# - https://stackoverflow.com/questions/16729574/how-can-i-get-a-value-from-a-cell-of-a-dataframe
# - https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
# - https://www.geeksforgeeks.org/create-a-pandas-dataframe-from-lists/
# - https://datagy.io/pandas-dataframe-to-list/#:~:text=In%20order%20to%20convert%20a%20Pandas%20DataFrame%20into%20a%20list,tolist()%20method.
# - https://stackoverflow.com/questions/2345151/how-to-save-read-class-wholly-in-python

def main():
    SEED, NUMBER_OF_MAXIMUM_CHANGES_PER_SAMPLE, HOW_MANY_NEW_SAMPLES,  = 623598, 5, 1000
    make_reproducible(SEED)
    real_sequences = get_real_sequences()
    real_and_mutated_sequences = add_mutations(real_sequences, HOW_MANY_NEW_SAMPLES, NUMBER_OF_MAXIMUM_CHANGES_PER_SAMPLE)
    save_real_and_mutated_sequences(real_and_mutated_sequences, SEED, NUMBER_OF_MAXIMUM_CHANGES_PER_SAMPLE, HOW_MANY_NEW_SAMPLES)
    real_and_mutated_sequences = read_real_and_mutated_sequences(SEED, NUMBER_OF_MAXIMUM_CHANGES_PER_SAMPLE, HOW_MANY_NEW_SAMPLES)
    laplace_results, rnn_results, transformer_results = evaluate_performance_for_different_models(real_and_mutated_sequences)
    print_evaluation_results(laplace_results, rnn_results, transformer_results)

def make_reproducible(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

def get_real_sequences():
    real_test_dataset = pd.read_csv('./actual_test_dataset.csv')
    real_sequences = real_test_dataset.drop(columns=['Symbol', 'Description', 'GeneType', 'GeneGroupIdentifier', 'GeneGroupMethod'])
    return real_sequences

def add_mutations(real_sequences, HOW_MANY_NEW_SAMPLES, NUMBER_OF_MAXIMUM_CHANGES_PER_SAMPLE):
    print('Mutated Dataset is being prepared.')
    count_samples, real_and_mutated_pairs = 0, []
    while count_samples < HOW_MANY_NEW_SAMPLES:
        real_and_mutated_pair = mutate_a_real_sequence(real_sequences, NUMBER_OF_MAXIMUM_CHANGES_PER_SAMPLE)
        real_and_mutated_pairs.append(real_and_mutated_pair)
        count_samples += 1
    print('Mutated Dataset was prepared.')
    return real_and_mutated_pairs

def mutate_a_real_sequence(real_sequences, NUMBER_OF_MAXIMUM_CHANGES_PER_SAMPLE):
    selected_sample = real_sequences.sample(n=1)
    selected_sequence = selected_sample['NucleotideSequence'].iloc[0]
    number_of_changes_for_selected_sample = np.random.randint(low=1, high=NUMBER_OF_MAXIMUM_CHANGES_PER_SAMPLE + 1)
    mutated_sequence = '<' + mutate_selected_sequence(selected_sequence[1:-1], number_of_changes_for_selected_sample) + '>'
    return (selected_sample['NCBIGeneID'], selected_sample['NucleotideSequence'].iloc[0], mutated_sequence)

def mutate_selected_sequence(selected_sequence, number_of_changes_for_selected_sample):
    count_changes, mutated_sequence = 0, selected_sequence
    while count_changes < number_of_changes_for_selected_sample:
        mutated_sequence = add_a_mutation(mutated_sequence)
        count_changes += 1
    return mutated_sequence

def add_a_mutation(sequence):
    chosen_mutation_type = np.random.randint(low=0, high=3 + 1)
    if chosen_mutation_type == 0:
        mutated_sequence = add_a_nucleotide(sequence)
    elif chosen_mutation_type == 1:
        mutated_sequence = delete_a_nucleotide(sequence)
    elif chosen_mutation_type == 2:
        mutated_sequence = swap_two_nucleotides(sequence)
    elif chosen_mutation_type == 3:
        mutated_sequence = change_a_nucleotide(sequence)
    return mutated_sequence

def add_a_nucleotide(sequence):
    nucleotide_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    position, which_nucleotide_to_add = np.random.randint(low=0, high=len(sequence) + 1), nucleotide_map[np.random.randint(low=0, high=3 + 1)]
    mutated_sequence = sequence[:position] + which_nucleotide_to_add + sequence[position:]
    return mutated_sequence

def delete_a_nucleotide(sequence):
    position = np.random.randint(low=0, high=len(sequence) - 1 + 1)
    mutated_sequence = sequence[:position] + sequence[position + 1:]
    return mutated_sequence

def swap_two_nucleotides(sequence):
    position = np.random.randint(low=0, high=len(sequence) - 2 + 1)
    while sequence[position] == sequence[position + 1]:
        position = np.random.randint(low=0, high=len(sequence) - 2 + 1)
    mutated_sequence = sequence[:position] + sequence[position + 1] + sequence[position] + sequence[position + 2:]
    return mutated_sequence

def change_a_nucleotide(sequence):
    nucleotide_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    position, which_nucleotide_to_put = np.random.randint(low=0, high=len(sequence) - 1 + 1), nucleotide_map[np.random.randint(low=0, high=3 + 1)]
    while sequence[position] == which_nucleotide_to_put:
        which_nucleotide_to_put = nucleotide_map[np.random.randint(low=0, high=3 + 1)]
    mutated_sequence = sequence[:position] + which_nucleotide_to_put + sequence[position + 1:]
    return mutated_sequence

def save_real_and_mutated_sequences(real_and_mutated_sequences, SEED, NUMBER_OF_MAXIMUM_CHANGES_PER_SAMPLE, HOW_MANY_NEW_SAMPLES):
    if not os.path.exists('./SyntheticDataset'):
        os.mkdir('./SyntheticDataset')

    formatted_real_and_mutated_sequences = format_real_and_mutated_sequences(real_and_mutated_sequences)
    csv_frame = pd.DataFrame(formatted_real_and_mutated_sequences, columns=['NCBIGeneID', 'RealNucleotideSequence', 'MutatedNucleotideSequence'])
    csv_frame.to_csv('./SyntheticDataset/synthetic_' + str(SEED) + '_' + str(NUMBER_OF_MAXIMUM_CHANGES_PER_SAMPLE) + '_' + str(HOW_MANY_NEW_SAMPLES) + '.csv', index=False)

def format_real_and_mutated_sequences(real_and_mutated_sequences):
    formatted_seq = []
    for i in range(len(real_and_mutated_sequences)):
        formatted_seq.append((real_and_mutated_sequences[i][0].tolist()[0], real_and_mutated_sequences[i][1], real_and_mutated_sequences[i][2]))
    return formatted_seq

def read_real_and_mutated_sequences(SEED, NUMBER_OF_MAXIMUM_CHANGES_PER_SAMPLE, HOW_MANY_NEW_SAMPLES):
    real_and_mutated_sequences = pd.read_csv('./SyntheticDataset/synthetic_' + str(SEED) + '_' + str(NUMBER_OF_MAXIMUM_CHANGES_PER_SAMPLE) + '_' + str(HOW_MANY_NEW_SAMPLES) + '.csv')
    real_and_mutated_sequences = real_and_mutated_sequences.to_records(index=False).tolist()

    return real_and_mutated_sequences

def evaluate_performance_for_different_models(real_and_mutated_sequences):
    laplace_results = evaluate_laplace_smoothing(real_and_mutated_sequences)
    rnn_results = evaluate_rnn(real_and_mutated_sequences)
    transformer_results = evaluate_transformer(real_and_mutated_sequences)
    return laplace_results, rnn_results, transformer_results

def evaluate_laplace_smoothing(real_and_mutated_sequences):
    print('Laplace Smoothing Model is evaluated.')
    n = 8
    lm = read_lm(n)
    print('Calculation Started.')
    perp_acc, prob_acc = get_performance_for_n_gram(n, lm, real_and_mutated_sequences)
    gc.collect()

    print('Calculation is completed. Perplexity accuracy value:', perp_acc)
    print('Calculation is completed. Probability accuracy value:', prob_acc)

    return (perp_acc, prob_acc)

def read_lm(n):
    print('LM Is Being Read.')
    f = open('./LaplaceModel/lang_model_s_' + str(n) + '.obj', 'rb')
    lm = pickle.load(f)
    print('LM Was Read.')
    return lm

def get_performance_for_n_gram(n, lm, real_and_mutated_sequences):
    gc.collect()
    start_time = time.time()
    correct_pred_number_perp, correct_pred_number_prob, wrong_pred_number_perp, wrong_pred_number_prob = 0, 0, 0, 0
    for i in range(len(real_and_mutated_sequences)):
        real_pred_perp, seq_len_real = evaluate_lang_model(lm, real_and_mutated_sequences[i][1], n)
        mutated_pred_perp, seq_len_mutated = evaluate_lang_model(lm, real_and_mutated_sequences[i][2], n)
        if real_pred_perp < mutated_pred_perp:
            correct_pred_number_perp += 1
        else:
            wrong_pred_number_perp += 1
        if is_probs_correct(real_pred_perp, seq_len_real, mutated_pred_perp, seq_len_mutated):
            correct_pred_number_prob += 1
        else:
            wrong_pred_number_prob += 1
        if i % 10 == 0:
            print(i, '/', len(real_and_mutated_sequences))
            print(correct_pred_number_perp / (correct_pred_number_perp + wrong_pred_number_perp))
            print(correct_pred_number_prob / (correct_pred_number_prob + wrong_pred_number_prob))
    print("--- %s seconds ---" % (time.time() - start_time))
    del lm
    gc.collect()

    return (correct_pred_number_perp / (correct_pred_number_perp + wrong_pred_number_perp)), \
           (correct_pred_number_prob / (correct_pred_number_prob + wrong_pred_number_prob))

def get_performance_for_rnn(model, real_and_mutated_sequences):
    gc.collect()
    BATCH_SIZE, SEQ_LEN, MIN_TRAINING_SEQ_LEN, VOCAB_SIZE = initialize_hyper_parameters()
    prepare_actual_dataset_prev('./vocab.csv', 'vocab')
    raw_train_ds = prepare_dataset(MIN_TRAINING_SEQ_LEN, BATCH_SIZE)
    vocab = get_vocab(raw_train_ds, VOCAB_SIZE)
    print('Vocabulary was prepared')
    start_time = time.time()
    correct_pred_number_perp, correct_pred_number_prob, wrong_pred_number_perp, wrong_pred_number_prob = 0, 0, 0, 0
    for i in range(len(real_and_mutated_sequences)):
        real_pred_perp, seq_len_real, seq_ds1 = evaluate_rnn_lang_model(model, real_and_mutated_sequences[i][1], vocab)
        mutated_pred_perp, seq_len_mutated, seq_ds2 = evaluate_rnn_lang_model(model, real_and_mutated_sequences[i][2], vocab)
        if real_pred_perp < mutated_pred_perp:
            correct_pred_number_perp += 1
        else:
            wrong_pred_number_perp += 1
        if is_probs_correct(real_pred_perp, seq_len_real, mutated_pred_perp, seq_len_mutated):
            correct_pred_number_prob += 1
        else:
            wrong_pred_number_prob += 1
        if i % 10 == 0:
            print(i, '/', len(real_and_mutated_sequences))
            print(correct_pred_number_perp / (correct_pred_number_perp + wrong_pred_number_perp))
            print(correct_pred_number_prob / (correct_pred_number_prob + wrong_pred_number_prob))
    print("--- %s seconds ---" % (time.time() - start_time))
    gc.collect()

    return (correct_pred_number_perp / (correct_pred_number_perp + wrong_pred_number_perp)), \
           (correct_pred_number_prob / (correct_pred_number_prob + wrong_pred_number_prob))

def get_performance_for_transformer(model, real_and_mutated_sequences):
    gc.collect()
    BATCH_SIZE, SEQ_LEN, MIN_TRAINING_SEQ_LEN, VOCAB_SIZE = initialize_hyper_parameters()
    prepare_actual_dataset_prev('./vocab.csv', 'vocab')
    raw_train_ds = prepare_dataset(MIN_TRAINING_SEQ_LEN, BATCH_SIZE)
    vocab = get_vocab(raw_train_ds, VOCAB_SIZE)
    print('Vocabulary was prepared')
    start_time = time.time()
    correct_pred_number_perp, correct_pred_number_prob, wrong_pred_number_perp, wrong_pred_number_prob = 0, 0, 0, 0
    for i in range(len(real_and_mutated_sequences)):
        real_pred_perp, seq_len_real, seq_ds1 = evaluate_transformer_lang_model(model, real_and_mutated_sequences[i][1], vocab)
        mutated_pred_perp, seq_len_mutated, seq_ds2 = evaluate_transformer_lang_model(model, real_and_mutated_sequences[i][2], vocab)

        if real_pred_perp < mutated_pred_perp:
            correct_pred_number_perp += 1
        else:
            wrong_pred_number_perp += 1
        if is_probs_correct(real_pred_perp, seq_len_real, mutated_pred_perp, seq_len_mutated):
            correct_pred_number_prob += 1
        else:
            wrong_pred_number_prob += 1
        if i % 10 == 0:
            print(i, '/', len(real_and_mutated_sequences))
            print(correct_pred_number_perp / (correct_pred_number_perp + wrong_pred_number_perp))
            print(correct_pred_number_prob / (correct_pred_number_prob + wrong_pred_number_prob))
    print("--- %s seconds ---" % (time.time() - start_time))
    gc.collect()

    return (correct_pred_number_perp / (correct_pred_number_perp + wrong_pred_number_perp)), \
           (correct_pred_number_prob / (correct_pred_number_prob + wrong_pred_number_prob))

def is_probs_correct(real_pred_perp, seq_len_real, mutated_pred_perp, seq_len_mutated):
    return (seq_len_real * np.log(real_pred_perp)) < (seq_len_mutated * np.log(mutated_pred_perp))

def get_formatted_data(gene_nucleotide_sequences_list):
    formatted_dataset = []
    for i in range(len(gene_nucleotide_sequences_list)):
        formatted_str = format_str(gene_nucleotide_sequences_list[i])
        formatted_dataset.append(formatted_str)
    del formatted_str
    gc.collect()

    return formatted_dataset

def format_str(gene_nucleotide_sequence):
    chars_list = [*gene_nucleotide_sequence[1:-1]]
    return chars_list

def evaluate_lang_model(lm, sequence, n, verbose=False):
    X_test = get_formatted_data([sequence])
    ranges, subresults = find_ranges(len(X_test), n_parts=50), []
    if verbose:
        print(ranges)
    for k in range(len(ranges)):
        X_test_preprocessed = []
        for i in range(ranges[k][0], min(ranges[k][1], len(X_test))):
            X_test_preprocessed += preprocess(X_test[i], n)
        if verbose:
            print('Formatting Completed. Perplexity Calculation Is Starting.', k, '/', len(ranges))
        perp = lm.perplexity(X_test_preprocessed)
        if verbose:
            print('Perplexity Was Calculated.', k, '/', len(ranges))
        subresults.append((len(X_test_preprocessed), perp))
        if verbose:
            print('N and P Values:', (len(X_test_preprocessed), perp))
        del X_test_preprocessed
        gc.collect()

    gc.collect()

    res = calculate_actual_perp(subresults)
    return res, len(sequence) - 1

def evaluate_rnn_lang_model(model, sequence, vocab, verbose=False):
    seq_ds = prepare_seq(sequence, vocab)
    res = do_inference(model, seq_ds)
    if verbose:
        print('Res Pair:', res)
    return res[1], len(sequence) - 1, seq_ds

def evaluate_transformer_lang_model(model, sequence, vocab, verbose=False):
    seq_ds = prepare_seq(sequence, vocab)
    res = do_inference(model, seq_ds)
    return res[1], len(sequence) - 1, seq_ds

def prepare_seq(sequence, vocab, verbose=False):
    BATCH_SIZE, SEQ_LEN, MIN_TRAINING_SEQ_LEN, VOCAB_SIZE = initialize_hyper_parameters()
    prepare_actual_dataset([sequence])
    raw_seq_ds = prepare_dataset_seq(MIN_TRAINING_SEQ_LEN, BATCH_SIZE)
    if verbose:
        print('Dataset was prepared')
    tokenizer = get_tokenizer(vocab, SEQ_LEN)
    if verbose:
        print('Tokenizer was prepared')
    seq_ds = tokenize_dataset(SEQ_LEN, tokenizer, raw_seq_ds)
    if verbose:
        print('Dataset was tokenized')
    return seq_ds

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

def tokenize_dataset(SEQ_LEN, tokenizer, raw_seq_ds):
    start_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=SEQ_LEN,
        start_value=tokenizer.token_to_id("[BOS]"),
    )

    def preprocess(inputs):
        outputs = tokenizer(inputs)
        features = start_packer(outputs)
        labels = outputs
        return features, labels

    seq_ds = raw_seq_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE
    )

    return seq_ds

def prepare_actual_dataset(gene_nucleotide_sequences_list):
    if not os.path.exists('./RNNModel/Datasets'):
        os.mkdir('./RNNModel/Datasets')

    f = open('./RNNModel/Datasets/' + 'sequence' + '.txt', 'w')
    for i in range(len(gene_nucleotide_sequences_list)):
        f.write(' '.join(gene_nucleotide_sequences_list[i][1:-1]))
        f.write('\n\n')

        if i % 100 == 3:
            print(i, '/', len(gene_nucleotide_sequences_list))
    f.close()

def prepare_actual_dataset_prev(path, name):
    dataframe = pd.read_csv(path)
    gene_nucleotide_sequences = dataframe['NucleotideSequence']
    gene_nucleotide_sequences_list = gene_nucleotide_sequences.tolist()

    if not os.path.exists('./RNNModel/Datasets'):
        os.mkdir('./RNNModel/Datasets')

    f = open('./RNNModel/Datasets/' + name + '.txt', 'w')
    for i in range(len(gene_nucleotide_sequences_list)):
        f.write(' '.join(gene_nucleotide_sequences_list[i][1:-1]))
        f.write('\n\n')

        if i % 100 == 3:
            print(i, '/', len(gene_nucleotide_sequences_list))
    f.close()

def prepare_dataset(MIN_TRAINING_SEQ_LEN, BATCH_SIZE):
    dir = os.path.expanduser('./RNNModel/Datasets/')

    raw_train_ds = (
        tf.data.TextLineDataset(dir + "vocab.txt")
            .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
            .batch(BATCH_SIZE)
            .shuffle(buffer_size=256)
    )

    return raw_train_ds

def prepare_dataset_seq(MIN_TRAINING_SEQ_LEN, BATCH_SIZE):
    dir = os.path.expanduser('./RNNModel/Datasets/')

    raw_seq_ds = (
        tf.data.TextLineDataset(dir + "sequence.txt")
            .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
            .batch(BATCH_SIZE)
            .shuffle(buffer_size=256)
    )

    return raw_seq_ds

def initialize_hyper_parameters():
    BATCH_SIZE = 64
    SEQ_LEN = 2**10 - 1
    MIN_TRAINING_SEQ_LEN = 2

    VOCAB_SIZE = 8

    return BATCH_SIZE, SEQ_LEN, MIN_TRAINING_SEQ_LEN, VOCAB_SIZE

def do_inference(model, ds):
    perp = model.evaluate(ds, verbose=0)
    return perp

def find_ranges(length_val, n_parts=10):
    part_size = (length_val // n_parts) + 1
    ranges_list = []
    count = 0
    while count < length_val:
        ranges_list.append((count, count+part_size))
        count += part_size
    return ranges_list

def preprocess(X_test, n):
    temp = pad_both_ends(X_test, n=n)
    res = list(ngrams(temp, n=n))
    if n > 2:
        res = res[:-(n-2)]
    return res

def calculate_actual_perp(subresults):
    n_vals, p_vals = [], []
    for i in range(len(subresults)):
        n_vals.append(subresults[i][0])
        p_vals.append(subresults[i][1])
    n_vals, p_vals = np.array(n_vals), np.array(p_vals)
    n_vals_sum = np.sum(n_vals)
    p_log_vals = np.log(p_vals)
    dividend = np.sum(n_vals * p_log_vals)
    return np.exp(dividend / n_vals_sum)

def evaluate_rnn(real_and_mutated_sequences):
    print('RNN Model is evaluated.')
    print('Model is being read')
    model = keras.models.load_model("./RNNModel/my_model")
    print('Model was read')
    print('Calculation Started.')
    perp_acc, prob_acc = get_performance_for_rnn(model, real_and_mutated_sequences)
    gc.collect()

    print('Calculation is completed. Perplexity accuracy value:', perp_acc)
    print('Calculation is completed. Probability accuracy value:', prob_acc)

    return (perp_acc, prob_acc)

def evaluate_transformer(real_and_mutated_sequences):
    print('Transformer Model is evaluated.')
    print('Model is being read')
    model = keras.models.load_model("./TransformerModel/my_model")
    print('Model was read')
    print('Calculation Started.')
    perp_acc, prob_acc = get_performance_for_transformer(model, real_and_mutated_sequences)
    gc.collect()

    print('Calculation is completed. Perplexity accuracy value:', perp_acc)
    print('Calculation is completed. Probability accuracy value:', prob_acc)

    return (perp_acc, prob_acc)

def print_evaluation_results(laplace_results, rnn_results, transformer_results):
    print()
    print('Results for Different Models:')
    print()
    print('Laplace Accuracy (using perplexity):\t\t', laplace_results[0])
    print('Laplace Accuracy (using probability):\t\t', laplace_results[1])
    print()
    print('RNN Accuracy (using perplexity):\t\t\t', rnn_results[0])
    print('RNN Accuracy (using probability):\t\t\t', rnn_results[1])
    print()
    print('Transformer Accuracy (using perplexity):\t', transformer_results[0])
    print('Transformer Accuracy (using probability):\t', transformer_results[1])

if __name__ == '__main__':
    main()
