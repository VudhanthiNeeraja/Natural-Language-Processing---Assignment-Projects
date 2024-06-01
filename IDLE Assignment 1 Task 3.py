Python 3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import nltk
>>> 
>>> sentences = nltk.corpus.brown.sents()
SyntaxError: invalid syntax
>>> sentences = nltk.corpus.brown.sents()
>>> import gensim.downloader as api
>>> wv = api.load("word2vec-google-news-300")
>>> wv.evaluate_word_pairs("tab-delimited.txt")
(PearsonRResult(statistic=0.35056250576058884, pvalue=0.3549934753074058), SignificanceResult(statistic=0.31666666666666665, pvalue=0.4063970144863861), 10.0)
>>> wv = KeyedVectors.load_word2vec_format("skipgram_embeddings.txt", binary=False)
Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    wv = KeyedVectors.load_word2vec_format("skipgram_embeddings.txt", binary=False)
NameError: name 'KeyedVectors' is not defined
>>> from gensim.models import KeyedVectors
>>> wv = KeyedVectors.load_word2vec_format("skipgram_embeddings.txt", binary=False)
>>> wv.evaluate_word_pairs("tab-delimited.txt")
(PearsonRResult(statistic=-0.34190613616445764, pvalue=0.45288517731850425), SignificanceResult(statistic=-0.1785714285714286, pvalue=0.7016579425162729), 30.0)
>>> 
KeyboardInterrupt
>>> wv = KeyedVectors.load_word2vec_format("cbow_embeddings.txt", binary=False)
>>> wv.evaluate_word_pairs("tab-delimited.txt")
(PearsonRResult(statistic=-0.43207089694664996, pvalue=0.33299866992205396), SignificanceResult(statistic=-0.03571428571428572, pvalue=0.9394082054712856), 30.0)
>>> 
