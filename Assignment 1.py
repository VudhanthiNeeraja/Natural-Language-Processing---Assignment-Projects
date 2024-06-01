Python 3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import nltk
>>> nltk.download()
showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
True
>>> sentences = nltk.corpus.brown.sents()
>>> len(sentences)
57340
>>> sentences[0]
['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', 'Friday', 'an', 'investigation', 'of', "Atlanta's", 'recent', 'primary', 'election', 'produced', '``', 'no', 'evidence', "''", 'that', 'any', 'irregularities', 'took', 'place', '.']
>>> type(sentences)
<class 'nltk.corpus.reader.util.ConcatenatedCorpusView'>
>>> sentences_list = list(sentences)

>>> type(sentences_list)
<class 'list'>
>>> sentences_list  = [[word.lower() for word in sent] for sent in sentences_list]
>>> lemmatizer = nltk.WordNetLemmatizer()
>>> sentences_list = [[lemmatizer.lemmatize(word) for word in sent] for sent in sentences_list]
>>> import gensim.downloader as api
>>> wv = api.load("word2vec-google-news-300")
>>> wv["king"]

>>> wv.similarity("king","queen")
0.6510956
>>> wv.most_similar("king")
[('kings', 0.7138045430183411), ('queen', 0.6510956883430481), ('monarch', 0.6413194537162781), ('crown_prince', 0.6204220056533813), ('prince', 0.6159993410110474), ('sultan', 0.5864824056625366), ('ruler', 0.5797567367553711), ('princes', 0.5646552443504333), ('Prince_Paras', 0.5432944297790527), ('throne', 0.5422105193138123)]
>>> v = wv["king"]-wv["male"]+wv["female"]
>>> wv.similar_by_vector(v)
[('king', 0.8830681443214417), ('queen', 0.6669611930847168), ('kings', 0.6140398979187012), ('monarch', 0.5949661135673523), ('crown_prince', 0.5778266787528992), ('sultan', 0.5558580160140991), ('ruler', 0.5497739911079407), ('prince', 0.5371752381324768), ('Savory_aromas_wafted', 0.5148698687553406), ('princess', 0.5138769149780273)]
>>> import gensim
>>> from gensim.models import Word2Vec
>>> import logging
>>> logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
>>> sg_params = {"vector_size": 100,"window": 5,"min_count": 1,"sg": 1,"epochs": 10 }
