Python 3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> #the five words I am using are the computer,pizza,ocean,rain,hat
>>> #first finding the most similar five words using pre-trained Google News embeddings
>>> import gensim.downloader as api
>>> wv = api.load("word2vec-google-news-300")
>>> list = wv.most_similar("computer", topn=5)
>>> print(list)
[('computers', 0.7979379892349243), ('laptop', 0.6640493273735046), ('laptop_computer', 0.6548868417739868), ('Computer', 0.647333562374115), ('com_puter', 0.6082080006599426)]
>>> list = wv.most_similar("pizza", topn=5)
>>> print(list)
[('pizzas', 0.7863470315933228), ('Domino_pizza', 0.7342829704284668), ('Pizza', 0.6988078355789185), ('pepperoni_pizza', 0.6902607083320618), ('sandwich', 0.6840401887893677)]
>>> list = wv.most_similar("ocean", topn=5)
>>> print(list)
[('sea', 0.7643541693687439), ('oceans', 0.7482994198799133), ('Pacific_Ocean', 0.7037094831466675), ('Atlantic_Ocean', 0.6659377217292786), ('oceanic', 0.6610181927680969)]
>>> list = wv.most_similar("rain", topn=5)
>>> print(list)
[('heavy_rain', 0.8421464562416077), ('downpour', 0.796761691570282), ('rains', 0.7827130556106567), ('torrential_rain', 0.7578904628753662), ('Rain', 0.7476006150245667)]
>>> list = wv.most_similar("hat", topn=5)
>>> print(list)
[('hats', 0.7639798521995544), ('straw_hat', 0.6072703003883362), ('cowboy_hat', 0.5864576101303101), ('Wearing_spangled', 0.5848298072814941), ('fedora', 0.5830994844436646)]
>>> #second finding the most similar five words using skipgram embeddings
>>> from gensim.models import KeyedVectors
>>> wv = KeyedVectors.load_word2vec_format("skipgram_embeddings.txt", binary=False)
>>> list = wv.most_similar("computer", topn=5)
>>> print(list)
[('digital', 0.8765760660171509), ('automation', 0.870654821395874), ('generate', 0.8688483238220215), ('packaged', 0.8658013939857483), ('injecting', 0.8652753233909607)]
>>> list = wv.most_similar("pizza", topn=5)
>>> print(list)
[('dine', 0.9492061734199524), ('mauch', 0.9457874894142151), ("probl'y", 0.9438025951385498), ('hating', 0.9428364038467407), ('hotdog', 0.9427337646484375)]
>>> list = wv.most_similar("ocean", topn=5)
>>> print(list)
[('basement', 0.8379862308502197), ('drainage', 0.8359599113464355), ('keel', 0.8224723935127258), ('stretching', 0.8180164098739624), ('masonry', 0.8179476857185364)]
>>> list = wv.most_similar("rain", topn=5)
>>> print(list)
[('mud', 0.8179415464401245), ('squall', 0.8155068159103394), ('wind', 0.8146893978118896), ('mist', 0.8075737953186035), ('storm', 0.8071972727775574)]
>>> list = wv.most_similar("hat", topn=5)
>>> print(list)
[('shirt', 0.8361958265304565), ('cap', 0.8165348172187805), ('pistol', 0.8153178095817566), ('boot', 0.811457097530365), ('forehead', 0.8098595142364502)]
>>> #second finding the most similar five words using cbow embeddings
>>> wv = KeyedVectors.load_word2vec_format("cbow_embeddings.txt", binary=False)
>>> list = wv.most_similar("computer", topn=5)
>>> print(list)
[('generating', 0.9132937788963318), ('installation', 0.9041382074356079), ('irregular', 0.896868109703064), ('reinforced', 0.8963931202888489), ('cleaning', 0.8885215520858765)]
>>> list = wv.most_similar("pizza", topn=5)
>>> print(list)
[('pro-trujillo', 0.9089798927307129), ('rag', 0.8928787708282471), ('soberly', 0.8883790373802185), ('vivian', 0.8805361390113831), ('get-together', 0.8795149326324463)]
>>> list = wv.most_similar("ocean", topn=5)
>>> print(list)
[('plug', 0.9035034775733948), ('chain', 0.8944990634918213), ('lamp', 0.8909759521484375), ('slope', 0.8890172243118286), ('blowing', 0.8852494359016418)]
>>> list = wv.most_similar("rain", topn=5)
>>> print(list)
[('storm', 0.8726685047149658), ('cool', 0.8587278127670288), ('bottle', 0.8579422831535339), ('grass', 0.8534609079360962), ('sun', 0.8533768653869629)]
>>> list = wv.most_similar("hat", topn=5)
>>> print(list)
[('lip', 0.9334967136383057), ('mouth', 0.899549126625061), ('hair', 0.8939570784568787), ('knee', 0.8880735635757446), ('beard', 0.8876423239707947)]
>>> 
