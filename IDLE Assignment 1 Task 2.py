Python 3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import nltk
>>> sentences = ntlk.corpus.brown.sents()
Traceback (most recent call last):
  File "<pyshell#1>", line 1, in <module>
    sentences = ntlk.corpus.brown.sents()
NameError: name 'ntlk' is not defined
>>> sentences = nltk.corpus.brown.sents()
>>> import gensim.downloader as api
>>> wv = api.load("word2vec-google-news-300")
>>> from gensim import models
>>> CBow = models.Word2Vecc(sentences,sg=0,epochs=3,vector_size=50,min_count=5)
Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    CBow = models.Word2Vecc(sentences,sg=0,epochs=3,vector_size=50,min_count=5)
AttributeError: module 'gensim.models' has no attribute 'Word2Vecc'
>>> CBow = models.Word2Vec(sentences,sg=0,epochs=3,vector_size=50,min_count=5)
>>> ModelVector = models.KeyedVectors.load_word2vec_format("cbow_vectors.txt",binary=False)
Traceback (most recent call last):
  File "<pyshell#8>", line 1, in <module>
    ModelVector = models.KeyedVectors.load_word2vec_format("cbow_vectors.txt",binary=False)
  File "C:\Users\Neeraja Vudhanthi\AppData\Local\Programs\Python\Python38\lib\site-packages\gensim\models\keyedvectors.py", line 1719, in load_word2vec_format
    return _load_word2vec_format(
  File "C:\Users\Neeraja Vudhanthi\AppData\Local\Programs\Python\Python38\lib\site-packages\gensim\models\keyedvectors.py", line 2048, in _load_word2vec_format
    with utils.open(fname, 'rb') as fin:
  File "C:\Users\Neeraja Vudhanthi\AppData\Local\Programs\Python\Python38\lib\site-packages\smart_open\smart_open_lib.py", line 177, in open
    fobj = _shortcut_open(
  File "C:\Users\Neeraja Vudhanthi\AppData\Local\Programs\Python\Python38\lib\site-packages\smart_open\smart_open_lib.py", line 363, in _shortcut_open
    return _builtin_open(local_path, mode, buffering=buffering, **open_kwargs)
FileNotFoundError: [Errno 2] No such file or directory: 'cbow_vectors.txt'
>>> ModelVector = models.KeyedVectors.load_word2vec_format("cbow_embeddings.txt",binary=False)
Traceback (most recent call last):
  File "<pyshell#9>", line 1, in <module>
    ModelVector = models.KeyedVectors.load_word2vec_format("cbow_embeddings.txt",binary=False)
  File "C:\Users\Neeraja Vudhanthi\AppData\Local\Programs\Python\Python38\lib\site-packages\gensim\models\keyedvectors.py", line 1719, in load_word2vec_format
    return _load_word2vec_format(
  File "C:\Users\Neeraja Vudhanthi\AppData\Local\Programs\Python\Python38\lib\site-packages\gensim\models\keyedvectors.py", line 2048, in _load_word2vec_format
    with utils.open(fname, 'rb') as fin:
  File "C:\Users\Neeraja Vudhanthi\AppData\Local\Programs\Python\Python38\lib\site-packages\smart_open\smart_open_lib.py", line 177, in open
    fobj = _shortcut_open(
  File "C:\Users\Neeraja Vudhanthi\AppData\Local\Programs\Python\Python38\lib\site-packages\smart_open\smart_open_lib.py", line 363, in _shortcut_open
    return _builtin_open(local_path, mode, buffering=buffering, **open_kwargs)
FileNotFoundError: [Errno 2] No such file or directory: 'cbow_embeddings.txt'
>>> ModelVector = models.KeyedVectors.load_word2vec_format("cbow_embeddings.txt",binary=False)
>>> impport A1_helper
SyntaxError: invalid syntax
>>> import A1_helper
>>> x_vals, y_vals, labels = A1_helper.reduce_dimensions(CBOW.wv)
Traceback (most recent call last):
  File "<pyshell#13>", line 1, in <module>
    x_vals, y_vals, labels = A1_helper.reduce_dimensions(CBOW.wv)
NameError: name 'CBOW' is not defined
>>> x_vals, y_vals, labels = A1_helper.reduce_dimensions(CBow.wv)
>>> A1_helper.plot_with_matplotlib(x_vals, y_vals, labels, ["I", "this", "red", "flag", "King", "Queen", "Math", "God", "English", "Food", "water", "eraser", "coins", "bookmark", "scales", "doctor", "help", "human", "told", "sea"])
Plotting I at 27.43317 31.411888
Plotting this at 34.435574 36.523838
Plotting red at 57.849873 37.74722
Plotting flag at -19.786985 -21.87393
Plotting King at 56.666035 11.947754
Plotting Queen at 42.826492 -31.825073
Math cannot be plotted because its word embedding is not given.
Plotting God at 34.604523 24.08503
Plotting English at 55.34289 39.878872
Plotting Food at -7.519597 -12.90356
Plotting water at 45.423214 44.50435
eraser cannot be plotted because its word embedding is not given.
Plotting coins at 1.3563762 -30.785421
bookmark cannot be plotted because its word embedding is not given.
Plotting scales at -38.51851 28.178768
Plotting doctor at 41.620834 -6.663023
Plotting help at 32.59338 20.152693
Plotting human at 46.831867 41.524063
Plotting told at 31.218168 25.130407
Plotting sea at 64.13028 27.414831
>>> Skip=models.Word2Vec(sentences,sg=1,epochs=3,vector_size=50,min_count=5)
>>> mv1=models.KeyedVectors.load_word2vec_format("skipgram_embeddings.txt",binary=False)
>>> x_vals, y_vals, labels = A1_helper.reduce_dimensions(Skip.wv)
>>> A1_helper.plot_with_matplotlib(x_vals, y_vals, labels, ["I", "this", "red", "flag", "King", "Queen", "Math", "God", "English", "Food", "water", "eraser", "coins", "bookmark", "scales", "doctor", "help", "human", "told", "sea"])
Plotting I at 38.226696 -59.58012
Plotting this at 65.74384 -30.494852
Plotting red at 37.151382 40.64658
Plotting flag at 23.567085 -6.919404
Plotting King at 5.9216166 63.384163
Plotting Queen at 15.67097 54.103985
Math cannot be plotted because its word embedding is not given.
Plotting God at 31.277729 -59.48015
Plotting English at 27.318317 55.946533
Plotting Food at -21.792368 56.892162
Plotting water at 38.77392 36.296883
eraser cannot be plotted because its word embedding is not given.
Plotting coins at -3.7448242 -27.5325
bookmark cannot be plotted because its word embedding is not given.
Plotting scales at -37.82182 -5.208837
Plotting doctor at 44.71564 -59.296803
Plotting help at 51.47649 -47.02374
Plotting human at 68.42725 -9.954746
Plotting told at 40.87521 -59.870975
Plotting sea at 43.61163 40.4784
>>> 
