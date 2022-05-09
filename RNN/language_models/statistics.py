import matplotlib.pyplot as plt
from d2l import tensorflow as d2l

tokens = d2l.tokenize(d2l.read_time_machine())
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)

frequencies = [freq for _, freq in vocab.token_freqs]
d2l.plot(frequencies, xlabel='token: x', ylabel='frequency : n(x)', xscale='log', yscale='log')

bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)

trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)

bigram_frequencies = [frequencies for _, frequencies in bigram_vocab.token_freqs]
trigram_frequencies = [frequencies for _, frequencies in trigram_vocab.token_freqs]
d2l.plot([frequencies,
          bigram_frequencies,
          trigram_frequencies],
         xlabel="token:x", ylabel="frequency : n(x)",
         xscale="log", yscale="log", legend=["unigram", "bigram", "trigram"])
plt.show()
