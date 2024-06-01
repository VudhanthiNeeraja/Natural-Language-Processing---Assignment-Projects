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
>>> skip_params = {"vector_size":150,"window":5,"min_count":1,"sg":1,"epochs":10}
>>> skipgram_model = Word2Vec(sentences=sentences_list, **cbow_params)
Traceback (most recent call last):
  File "<pyshell#23>", line 1, in <module>
    skipgram_model = Word2Vec(sentences=sentences_list, **cbow_params)
NameError: name 'cbow_params' is not defined
>>> skipgram_model = Word2Vec(sentences=sentences_list, **skip_params)
2023-10-17 16:51:09,501 : INFO : collecting all words and their counts
2023-10-17 16:51:09,548 : INFO : PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
2023-10-17 16:51:09,611 : INFO : PROGRESS: at sentence #10000, processed 219770 words, keeping 19060 word types
2023-10-17 16:51:09,673 : INFO : PROGRESS: at sentence #20000, processed 430477 words, keeping 27401 word types
2023-10-17 16:51:09,816 : INFO : PROGRESS: at sentence #30000, processed 669056 words, keeping 33652 word types
2023-10-17 16:51:09,942 : INFO : PROGRESS: at sentence #40000, processed 888291 words, keeping 39023 word types
2023-10-17 16:51:10,068 : INFO : PROGRESS: at sentence #50000, processed 1039920 words, keeping 42087 word types
2023-10-17 16:51:10,195 : INFO : collected 44539 word types from a corpus of 1161192 raw words and 57340 sentences
2023-10-17 16:51:10,211 : INFO : Creating a fresh vocabulary
2023-10-17 16:51:10,462 : INFO : Word2Vec lifecycle event {'msg': 'effective_min_count=1 retains 44539 unique words (100.00% of original 44539, drops 0)', 'datetime': '2023-10-17T16:51:10.462661', 'gensim': '4.3.2', 'python': '3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.22621-SP0', 'event': 'prepare_vocab'}
2023-10-17 16:51:10,478 : INFO : Word2Vec lifecycle event {'msg': 'effective_min_count=1 leaves 1161192 word corpus (100.00% of original 1161192, drops 0)', 'datetime': '2023-10-17T16:51:10.478287', 'gensim': '4.3.2', 'python': '3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.22621-SP0', 'event': 'prepare_vocab'}
2023-10-17 16:51:10,762 : INFO : deleting the raw counts dictionary of 44539 items
2023-10-17 16:51:10,763 : INFO : sample=0.001 downsamples 40 most-common words
2023-10-17 16:51:10,763 : INFO : Word2Vec lifecycle event {'msg': 'downsampling leaves estimated 831328.8287296845 word corpus (71.6%% of prior 1161192)', 'datetime': '2023-10-17T16:51:10.763472', 'gensim': '4.3.2', 'python': '3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.22621-SP0', 'event': 'prepare_vocab'}
2023-10-17 16:51:11,157 : INFO : estimated required memory for 44539 words and 150 dimensions: 75716300 bytes
2023-10-17 16:51:11,157 : INFO : resetting layer weights
2023-10-17 16:51:11,192 : INFO : Word2Vec lifecycle event {'update': False, 'trim_rule': 'None', 'datetime': '2023-10-17T16:51:11.192943', 'gensim': '4.3.2', 'python': '3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.22621-SP0', 'event': 'build_vocab'}
2023-10-17 16:51:11,205 : INFO : Word2Vec lifecycle event {'msg': 'training model with 3 workers on 44539 vocabulary and 150 features, using sg=1 hs=0 sample=0.001 negative=5 window=5 shrink_windows=True', 'datetime': '2023-10-17T16:51:11.205450', 'gensim': '4.3.2', 'python': '3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.22621-SP0', 'event': 'train'}
2023-10-17 16:51:12,232 : INFO : EPOCH 0 - PROGRESS: at 31.03% examples, 269634 words/s, in_qsize 5, out_qsize 0
2023-10-17 16:51:13,229 : INFO : EPOCH 0 - PROGRESS: at 53.82% examples, 244778 words/s, in_qsize 5, out_qsize 0
2023-10-17 16:51:14,239 : INFO : EPOCH 0 - PROGRESS: at 89.23% examples, 250314 words/s, in_qsize 5, out_qsize 0
2023-10-17 16:51:14,496 : INFO : EPOCH 0: training on 1161192 raw words (831312 effective words) took 3.3s, 253075 effective words/s
2023-10-17 16:51:15,534 : INFO : EPOCH 1 - PROGRESS: at 33.41% examples, 292953 words/s, in_qsize 6, out_qsize 0
2023-10-17 16:51:16,560 : INFO : EPOCH 1 - PROGRESS: at 58.31% examples, 265029 words/s, in_qsize 6, out_qsize 0
2023-10-17 16:51:17,556 : INFO : EPOCH 1 - PROGRESS: at 100.00% examples, 274121 words/s, in_qsize 0, out_qsize 1
2023-10-17 16:51:17,572 : INFO : EPOCH 1: training on 1161192 raw words (831546 effective words) took 3.0s, 272925 effective words/s
2023-10-17 16:51:18,602 : INFO : EPOCH 2 - PROGRESS: at 34.02% examples, 296675 words/s, in_qsize 5, out_qsize 0
2023-10-17 16:51:19,626 : INFO : EPOCH 2 - PROGRESS: at 64.67% examples, 295913 words/s, in_qsize 6, out_qsize 0
2023-10-17 16:51:20,466 : INFO : EPOCH 2: training on 1161192 raw words (831363 effective words) took 2.9s, 288432 effective words/s
2023-10-17 16:51:21,542 : INFO : EPOCH 3 - PROGRESS: at 34.02% examples, 298459 words/s, in_qsize 5, out_qsize 0
2023-10-17 16:51:22,552 : INFO : EPOCH 3 - PROGRESS: at 67.44% examples, 308342 words/s, in_qsize 5, out_qsize 0
2023-10-17 16:51:23,167 : INFO : EPOCH 3: training on 1161192 raw words (831502 effective words) took 2.6s, 315458 effective words/s
2023-10-17 16:51:24,211 : INFO : EPOCH 4 - PROGRESS: at 34.78% examples, 304558 words/s, in_qsize 6, out_qsize 0
2023-10-17 16:51:25,210 : INFO : EPOCH 4 - PROGRESS: at 61.85% examples, 283803 words/s, in_qsize 5, out_qsize 0
2023-10-17 16:51:25,982 : INFO : EPOCH 4: training on 1161192 raw words (831565 effective words) took 2.8s, 297594 effective words/s
2023-10-17 16:51:27,042 : INFO : EPOCH 5 - PROGRESS: at 34.78% examples, 294662 words/s, in_qsize 5, out_qsize 0
2023-10-17 16:51:28,086 : INFO : EPOCH 5 - PROGRESS: at 65.57% examples, 290342 words/s, in_qsize 5, out_qsize 0
2023-10-17 16:51:28,702 : INFO : EPOCH 5: training on 1161192 raw words (831274 effective words) took 2.7s, 306827 effective words/s
2023-10-17 16:51:29,715 : INFO : EPOCH 6 - PROGRESS: at 35.48% examples, 316274 words/s, in_qsize 5, out_qsize 0
2023-10-17 16:51:30,742 : INFO : EPOCH 6 - PROGRESS: at 70.94% examples, 317964 words/s, in_qsize 6, out_qsize 0
2023-10-17 16:51:31,408 : INFO : EPOCH 6: training on 1161192 raw words (831147 effective words) took 2.7s, 309864 effective words/s
2023-10-17 16:51:32,419 : INFO : EPOCH 7 - PROGRESS: at 32.60% examples, 287142 words/s, in_qsize 5, out_qsize 0
2023-10-17 16:51:33,431 : INFO : EPOCH 7 - PROGRESS: at 63.18% examples, 290607 words/s, in_qsize 6, out_qsize 0
2023-10-17 16:51:34,143 : INFO : EPOCH 7: training on 1161192 raw words (831358 effective words) took 2.7s, 304886 effective words/s
2023-10-17 16:51:35,203 : INFO : EPOCH 8 - PROGRESS: at 34.02% examples, 302104 words/s, in_qsize 5, out_qsize 0
2023-10-17 16:51:36,199 : INFO : EPOCH 8 - PROGRESS: at 67.44% examples, 310588 words/s, in_qsize 6, out_qsize 0
2023-10-17 16:51:36,831 : INFO : EPOCH 8: training on 1161192 raw words (831385 effective words) took 2.6s, 315234 effective words/s
2023-10-17 16:51:37,873 : INFO : EPOCH 9 - PROGRESS: at 34.78% examples, 304274 words/s, in_qsize 6, out_qsize 0
2023-10-17 16:51:38,914 : INFO : EPOCH 9 - PROGRESS: at 67.44% examples, 302563 words/s, in_qsize 6, out_qsize 0
2023-10-17 16:51:39,530 : INFO : EPOCH 9: training on 1161192 raw words (831489 effective words) took 2.7s, 309553 effective words/s
2023-10-17 16:51:39,545 : INFO : Word2Vec lifecycle event {'msg': 'training on 11611920 raw words (8313941 effective words) took 28.3s, 293391 effective words/s', 'datetime': '2023-10-17T16:51:39.545742', 'gensim': '4.3.2', 'python': '3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.22621-SP0', 'event': 'train'}
2023-10-17 16:51:39,545 : INFO : Word2Vec lifecycle event {'params': 'Word2Vec<vocab=44539, vector_size=150, alpha=0.025>', 'datetime': '2023-10-17T16:51:39.545742', 'gensim': '4.3.2', 'python': '3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.22621-SP0', 'event': 'created'}
>>> skipgram_model.wv.save_word2vec_format("skipgram_embeddings.txt", binary=False)
2023-10-17 16:52:40,813 : INFO : storing 44539x150 projection weights into skipgram_embeddings.txt
>>> cbow_params = {"vector_size": 100, "window": 5, "min_count":1, "sg":0, "epoch":10}
>>> cbow_model = Word2Vec(sentences = sentences_list, **cbow_params)
Traceback (most recent call last):
  File "<pyshell#27>", line 1, in <module>
    cbow_model = Word2Vec(sentences = sentences_list, **cbow_params)
TypeError: __init__() got an unexpected keyword argument 'epoch'
>>> cbow_params = {"vector_size": 100, "window": 5, "min_count":1, "sg":0, "epochs":10}
>>> cbow_model = Word2Vec(sentences = sentences_list, **cbow_params)
2023-10-17 16:57:30,635 : INFO : collecting all words and their counts
2023-10-17 16:57:30,683 : INFO : PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
2023-10-17 16:57:30,751 : INFO : PROGRESS: at sentence #10000, processed 219770 words, keeping 19060 word types
2023-10-17 16:57:30,808 : INFO : PROGRESS: at sentence #20000, processed 430477 words, keeping 27401 word types
2023-10-17 16:57:30,871 : INFO : PROGRESS: at sentence #30000, processed 669056 words, keeping 33652 word types
2023-10-17 16:57:31,014 : INFO : PROGRESS: at sentence #40000, processed 888291 words, keeping 39023 word types
2023-10-17 16:57:31,110 : INFO : PROGRESS: at sentence #50000, processed 1039920 words, keeping 42087 word types
2023-10-17 16:57:31,140 : INFO : collected 44539 word types from a corpus of 1161192 raw words and 57340 sentences
2023-10-17 16:57:31,152 : INFO : Creating a fresh vocabulary
2023-10-17 16:57:31,314 : INFO : Word2Vec lifecycle event {'msg': 'effective_min_count=1 retains 44539 unique words (100.00% of original 44539, drops 0)', 'datetime': '2023-10-17T16:57:31.314575', 'gensim': '4.3.2', 'python': '3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.22621-SP0', 'event': 'prepare_vocab'}
2023-10-17 16:57:31,314 : INFO : Word2Vec lifecycle event {'msg': 'effective_min_count=1 leaves 1161192 word corpus (100.00% of original 1161192, drops 0)', 'datetime': '2023-10-17T16:57:31.314575', 'gensim': '4.3.2', 'python': '3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.22621-SP0', 'event': 'prepare_vocab'}
2023-10-17 16:57:31,660 : INFO : deleting the raw counts dictionary of 44539 items
2023-10-17 16:57:31,676 : INFO : sample=0.001 downsamples 40 most-common words
2023-10-17 16:57:31,691 : INFO : Word2Vec lifecycle event {'msg': 'downsampling leaves estimated 831328.8287296845 word corpus (71.6%% of prior 1161192)', 'datetime': '2023-10-17T16:57:31.691910', 'gensim': '4.3.2', 'python': '3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.22621-SP0', 'event': 'prepare_vocab'}
2023-10-17 16:57:32,164 : INFO : estimated required memory for 44539 words and 100 dimensions: 57900700 bytes
2023-10-17 16:57:32,180 : INFO : resetting layer weights
2023-10-17 16:57:32,211 : INFO : Word2Vec lifecycle event {'update': False, 'trim_rule': 'None', 'datetime': '2023-10-17T16:57:32.211616', 'gensim': '4.3.2', 'python': '3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.22621-SP0', 'event': 'build_vocab'}
2023-10-17 16:57:32,227 : INFO : Word2Vec lifecycle event {'msg': 'training model with 3 workers on 44539 vocabulary and 100 features, using sg=0 hs=0 sample=0.001 negative=5 window=5 shrink_windows=True', 'datetime': '2023-10-17T16:57:32.227221', 'gensim': '4.3.2', 'python': '3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.22621-SP0', 'event': 'train'}
2023-10-17 16:57:32,938 : INFO : EPOCH 0: training on 1161192 raw words (831533 effective words) took 0.7s, 1187311 effective words/s
2023-10-17 16:57:33,813 : INFO : EPOCH 1: training on 1161192 raw words (831484 effective words) took 0.9s, 975007 effective words/s
2023-10-17 16:57:34,470 : INFO : EPOCH 2: training on 1161192 raw words (831202 effective words) took 0.7s, 1257260 effective words/s
2023-10-17 16:57:35,155 : INFO : EPOCH 3: training on 1161192 raw words (831556 effective words) took 0.7s, 1234829 effective words/s
2023-10-17 16:57:35,819 : INFO : EPOCH 4: training on 1161192 raw words (831166 effective words) took 0.7s, 1257831 effective words/s
2023-10-17 16:57:36,528 : INFO : EPOCH 5: training on 1161192 raw words (831233 effective words) took 0.7s, 1201707 effective words/s
2023-10-17 16:57:37,336 : INFO : EPOCH 6: training on 1161192 raw words (830949 effective words) took 0.7s, 1114068 effective words/s
2023-10-17 16:57:37,988 : INFO : EPOCH 7: training on 1161192 raw words (831534 effective words) took 0.7s, 1273598 effective words/s
2023-10-17 16:57:38,880 : INFO : EPOCH 8: training on 1161192 raw words (831236 effective words) took 0.9s, 951376 effective words/s
2023-10-17 16:57:39,768 : INFO : EPOCH 9: training on 1161192 raw words (831378 effective words) took 0.9s, 938411 effective words/s
2023-10-17 16:57:39,784 : INFO : Word2Vec lifecycle event {'msg': 'training on 11611920 raw words (8313271 effective words) took 7.5s, 1102277 effective words/s', 'datetime': '2023-10-17T16:57:39.784396', 'gensim': '4.3.2', 'python': '3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.22621-SP0', 'event': 'train'}
2023-10-17 16:57:39,784 : INFO : Word2Vec lifecycle event {'params': 'Word2Vec<vocab=44539, vector_size=100, alpha=0.025>', 'datetime': '2023-10-17T16:57:39.784396', 'gensim': '4.3.2', 'python': '3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.22621-SP0', 'event': 'created'}
>>> cbow_model.wv.save_word2vec_format("cbow_embeddings.txt", binary=False)
2023-10-17 16:58:14,830 : INFO : storing 44539x100 projection weights into cbow_embeddings.txt
>>> 
