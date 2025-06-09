'''
Input is going to be a sentence of type string

Algorithm:

First step is to compute the frequency table of the corpus
    - Represent this as a dict[tuple[bytes], int], {(l, o, w): 5}

    - Merge step
        - calculate each pair of tokens frequency and get the greatest one
            - Use max() to get the lexicographically greatest one
            - Optimization step for only incrementing the counts for pairs that just got merged instead of recomputing every merge
        - Add the greatest occurrence pair to the vocab
        - Repeat x number of times or until no more possible merging can happen

    - return
'''
import regex as re
import os
from typing import BinaryIO
from collections import Counter
from multiprocessing import Pool, Manager

# A regex pattern to split text into tokens, following the style of GPT-2/3/4.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:

    def __init__(self, input_path, vocab_size, special_tokens):
        self.vocab = {}
        self.id_to_token = {}
        self.freq_corpus = {}
        self.pair_counts = Counter()
        self.merges = []
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def initialize_vocab(self):
        self.vocab = {bytes([i]): i for i in range(256)}
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        next_id = 256
        for st_str in self.special_tokens:
            st_bytes = st_str.encode("utf-8")
            if st_bytes not in self.vocab:
                self.vocab[st_bytes] = next_id
                self.id_to_token[next_id] = st_bytes
                next_id += 1

    def calculate_freq_corpus(self):
        with open(self.input_path, 'r', encoding='utf-8') as f:
            text_data = f.read()

        tokens = re.findall(PAT, text_data)
        byte_tuples = [tuple(token.encode("utf-8")) for token in tokens]
        self.freq_corpus = Counter(byte_tuples)
        print(self.freq_corpus)

    def calculate_pair_counts(self):
        for word_ids, freq in self.freq_corpus.items():
            for i in range(len(word_ids) - 1):
                pair = (word_ids[i], word_ids[i + 1])
                self.pair_counts[pair] += freq

    def get_frequent_pair(self):
        if not self.pair_counts:
            return None
        return max(self.pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

    def merge_pair_in_corpus(self, pair_to_merge):
        new_freq_corpus = Counter()
        id1, id2 = pair_to_merge

        token1_bytes = self.id_to_token[id1]
        token2_bytes = self.id_to_token[id2]
        new_token_bytes = token1_bytes + token2_bytes

        new_id = len(self.vocab)
        self.vocab[new_token_bytes] = new_id
        self.id_to_token[new_id] = new_token_bytes
        self.merges.append((token1_bytes, token2_bytes))

        for word_ids, freq in self.freq_corpus.items():
            new_word_ids = []
            i = 0
            while i < len(word_ids):
                if i < len(word_ids) - 1 and word_ids[i] == id1 and word_ids[i + 1] == id2:
                    new_word_ids.append(new_id)
                    i += 2
                else:
                    new_word_ids.append(word_ids[i])
                    i += 1
            new_freq_corpus[tuple(new_word_ids)] += freq
        self.freq_corpus = new_freq_corpus

    def pretokenization(self):
        self.initialize_vocab()
        self.calculate_freq_corpus()

    def train(self):
        while self.vocab_size > len(self.vocab):
            self.pair_counts = Counter()
            self.calculate_pair_counts()
            most_freq_pair = self.get_frequent_pair()
            if not most_freq_pair:
                print("No more pairs to merge.")
                break
            token1 = self.id_to_token[most_freq_pair[0]].decode('utf-8', errors='ignore')
            token2 = self.id_to_token[most_freq_pair[1]].decode('utf-8', errors='ignore')
            print((token1, token2))

            self.merge_pair_in_corpus(most_freq_pair)

        print(self.vocab)
        print(self.merges)

    def find_chunk_boundaries(self, file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
        assert isinstance(split_special_token, bytes), (
            "Must represent special token as a bytestring"
        )

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def pretokenize_and_train_file(self):
        with open(self.input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, 10, "<|endoftext|>".encode("utf-8"))
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                self.pretokenization()
        self.train()


test_bpe_tokenizer = BPETokenizer("data\TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
test_bpe_tokenizer.pretokenize_and_train_file()
