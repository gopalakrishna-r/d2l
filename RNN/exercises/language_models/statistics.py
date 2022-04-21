import random
import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt
import re
import os, hashlib, requests, json

# Suppose there are 10000 words in the training dataset.
# How much word frequency and multi-word adjacent frequency does a four-gram need to store?

DATA_HUB = dict()
DATA_HUB['stupid_stuff'] = ('https://raw.githubusercontent.com/taivop/joke-dataset/master/' + 'stupidstuff.json')


def download_json(name, cache_dir=os.path.join('..', 'data')):  # @save
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url = DATA_HUB[name]
    d2l.mkdir_if_not_exist(cache_dir)
    filename = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
    print(f'Downloading {filename} from {url}...')
    with open(filename, 'w', encoding="utf-8") as f:
        r = requests.get(url, stream=True, verify=True)
        data = r.json()
        for j_body in data:
            f.write(j_body['body'])
            f.write('\n')
    return filename


def read_dataset():
    """Load the stupid stuff joke dataset into a list of text lines."""
    with open(download_json('stupid_stuff'), 'r', encoding="utf-8") as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


tokens = d2l.tokenize(read_dataset())
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)

frequencies = [freq for _, freq in vocab.token_freqs]

tetragram_tokens = [tuple for tuple in zip(corpus[:-3], corpus[1:-1], corpus[2:], corpus[3:])]
tetragram_vocab = d2l.Vocab(tetragram_tokens)

tetragram_frequencies = [frequencies for _, frequencies in tetragram_vocab.token_freqs]
d2l.plot([frequencies,
          tetragram_frequencies],
         xlabel="token:x", ylabel="frequency : n(x)",
         xscale="log", yscale="log", legend=["unigram", "tetragram"])
plt.show()
