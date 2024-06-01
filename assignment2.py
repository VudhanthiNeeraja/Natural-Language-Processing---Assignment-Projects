import gensim
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Bidirectional, Dense, TimeDistributed
from tensorflow.keras.initializers import Constant
from tensorflow.keras.utils import to_categorical
def make_map(sent):
    m = {}
    reverse_m = []
    m["_PAD_"]=0
    reverse_m.append("_PAD_")
    m["_UNK_"]=1
    reverse_m.append("_UNK_")
    i=2
    for s in sent:
        for w in s:
            if not w in m:
                m[w] = i
                reverse_m.append(w)
                i+=1
    return m,reverse_m
def read_data():
    import csv
    infile = open("C:/Users/Neeraja Vudhanthi/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Python 3.10/ner_dataset.csv","r")
    data = csv.reader(infile)
    lines = [w for w in data]
    infile.close()
    sent = []
    pos = []
    ner = []
    c = 0
    for i in range(1, len(lines)):
        d = lines[i]
        if "Sentence" in d[0]:
            if c>0:
                sent.append(s)
                pos.append(p)
                ner.append(n)
            s = [d[1]]
            p = [d[2]]
            n = [d[3]]
            c += 1
        else:
            s.append(d[1])
            p.append(d[2])
            n.append(d[3])
    sent.append(s)
    pos.append(p)
    ner.append(n)
    return sent,pos,ner
def get_embedding_matrix(wordmap):
    import gensim.downloader as api
    wv = api.load("word2vec-google-news-300")
    embedding_matrix = np.zeros((len(wordmap),300))
    for w in wordmap:
        if w in wv:
            embedding_matrix[wordmap[w]] = wv[w]
    return embedding_matrix
def vectorize_sent(sent,wordmap,maxlen):
    v_sent = np.zeros((len(sent),maxlen))
    for i in range(len(sent)) :
        for j in range(min(len(sent[i]),maxlen)):
            v_sent[i][j] = wordmap.get(sent[i][j],1)
    return v_sent
def tag_sentences(sent, m, wordmap, reverse_tagmap, maxlen) :
    v_sent = vectorize_sent(sent,wordmap,maxlen)
    predictions = m.predict(v_sent) 
    all_tags = [] 
    for i in range(len(predictions)) :
        num_p = [np.argmax(w) for w in predictions[i]]
        tags = [reverse_tagmap[p] for p in num_p]
        if len(tags) > len(sent[i]) :
            tags = tags[:len(sent[i])] 
        elif len(tags) < len(sent[i]) : 
            tags += ["O"]*(len(sent[i])-len(tags)) 
        all_tags.append(tags)
    return all_tags
def give_entities(sentence, tags) :
    assert(len(sentence)==len(tags))
    r = [""]*len(sentence)
    i = 0
    while i < len(sentence) :
        if tags[i][0]=="B" : # Beginning of the entity detected
            etype = tags[i][2:]
            name = sentence[i]
            start = i
            i += 1
            while i < len(sentence) :
                if tags[i]=="I-"+etype :
                    name += " "+sentence[i]
                    i += 1
                elif tags[i]=="O" :
                    i += 1
                else :
                    break
            r[start] = (etype,name)
        else :
            i += 1
    return r
def evaluate_sentence(s,target,predicted,entity_type_eval,outfile) :
    assert(len(s)==len(target))
    assert(len(s)==len(predicted))
    print("Sentence:",s,file=outfile)
    print("Target:",target,file=outfile)
    print("Predicted:",predicted,"\n",file=outfile)
    target_entities = give_entities(s,target)
    predicted_entities = give_entities(s,predicted)
    matched = [False]*len(s)
    for i in range(len(s)) :
        if target_entities[i] != "" :
            etype = target_entities[i][0]
            entity_type_eval[etype][0] += 1
            print("At",i,target_entities[i],end=" ",file=outfile)
            if target_entities[i]==predicted_entities[i] :
                matched[i] = True
                entity_type_eval[etype][1] += 1
                entity_type_eval[etype][2] += 1
                print("Extracted.",file=outfile)
            else :
                print("Missed.",file=outfile)
    for i in range(len(s)) :
        if (not matched[i]) and predicted_entities[i] != "" :
            etype = predicted_entities[i][0]
            entity_type_eval[etype][1] += 1
            print("At",i,predicted_entities[i],end=" ",file=outfile)
            print("Incorrectly extracted.",file=outfile)
def evaluate_ie(sent, target, predicted, entity_types, eval_filename) :
    assert(len(sent)==len(target))
    assert(len(sent)==len(predicted))
    entity_type_eval = {}
    for e in entity_types :
        entity_type_eval[e] = [0,0,0]
    outfile = open(eval_filename,"w")
    for i in range(len(sent)) :
        print("-------------------------",file=outfile)
        print("Test example:",i,file=outfile)
        evaluate_sentence(sent[i],target[i],predicted[i],entity_type_eval,outfile)
    print("\n\nEvaluation results:\n",file=outfile)
    for e in entity_types :
        print("\nEntity type:",e,file=outfile)
        print("Total entities:",entity_type_eval[e][0],file=outfile)
        print("Total predicted:",entity_type_eval[e][1],file=outfile)
        print("Correctly extracted:",entity_type_eval[e][2],file=outfile)
        if (entity_type_eval[e][1] > 0) :
            precision = (100*entity_type_eval[e][2])/entity_type_eval[e][1]
            print("Precision:",round(precision,2),"%",file=outfile)
        else :
            print("Precision cannot be computed.",file=outfile)
        if (entity_type_eval[e][0] > 0) :
            recall = (100*entity_type_eval[e][2])/entity_type_eval[e][0]
            print("Recall:",round(recall,2),"%",file=outfile)
        else :
            print("Recall cannot be computed.",file=outfile)
        if (entity_type_eval[e][1] > 0) and (entity_type_eval[e][0] > 0) :
            if precision+recall > 0 :
                f_measure = (2*precision*recall)/(precision+recall)
                print("F-measure:",round(f_measure,2),"%",file=outfile)
            else :
                print("F-measure cannot be computed.",file=outfile)
        else :
            print("F-measure cannot be computed.",file=outfile)
    all_total_entities = 0
    all_total_predicted = 0
    all_correct = 0
    for e in entity_types :
        all_total_entities += entity_type_eval[e][0]
        all_total_predicted += entity_type_eval[e][1]
        all_correct += entity_type_eval[e][2]
    print("\n\nAll entities combined:\n",file=outfile)
    print("Total entities:",all_total_entities,file=outfile)
    print("Total predicted:",all_total_predicted,file=outfile)
    print("Correctly extracted:",all_correct,file=outfile)
    if (all_total_predicted > 0) :
        precision = (100*all_correct)/all_total_predicted
        print("Precision:",round(precision,2),"%",file=outfile)
    else :
        print("Precision cannot be computed.",file=outfile)
    if (all_total_entities > 0) :
        recall = (100*all_correct)/all_total_entities
        print("Recall:",round(recall,2),"%",file=outfile)
    else :
        print("Recall cannot be computed.",file=outfile)
    if (all_total_predicted > 0) and (all_total_entities > 0) :
        if precision+recall > 0 :
            f_measure = (2*precision*recall)/(precision+recall)
            print("F-measure:",round(f_measure,2),"%",file=outfile)
        else :
            print("F-measure cannot be computed.",file=outfile)
    else :
        print("F-measure cannot be computed.",file=outfile)
    outfile.close()

def main():
    sent,pos,ner = read_data()
    wordmap, reverse_wordmap = make_map(sent)
    print("Number of unique words:", len(wordmap))
    vocab_len = len(wordmap)
    tagmap, reverse_tagmap = make_map(ner)
    print("Number of unique tags:", len(tagmap))
    tag_len = len(tagmap)
    entity_types = []
    for p in tagmap:
        if len(p) > 2:
            if p[:2] == "B-":
                entity_types.append(p[2:])
    print("Entity")
    print("Entity types:",entity_types)
    tr_sent = sent[:40000]
    tr_tag = ner[:40000]
    te_sent = sent[40000:]
    te_tag = ner[40000:]
    maxlen = 30
    print("Maximum sequence length set to:",maxlen)
    v_tr_sent = vectorize_sent(tr_sent,wordmap,maxlen)
    v_tr_target = vectorize_sent(tr_tag,tagmap,maxlen)
    v_tr_target = to_categorical(v_tr_target,num_classes=tag_len)
    embedding_matrix = get_embedding_matrix(wordmap)
    embed_len = 300
    print(v_tr_sent.shape)
    # Creating RNNs for Comparison: Different number of layers and Different number of nodes
    m1 = Sequential()
    m1.add(Embedding(vocab_len, embed_len,
                    embeddings_initializer=Constant(embedding_matrix)))
    m1.add(Bidirectional(LSTM(10, return_sequences=True, activation="relu")))
    m1.add(Bidirectional(LSTM(10, return_sequences=True, activation="relu")))
    m1.add(TimeDistributed(Dense(50, activation="relu")))
    m1.add(Dense(tag_len, activation="softmax"))
    m1.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    m2 = Sequential()
    m2.add(Embedding(vocab_len, embed_len,
                    embeddings_initializer=Constant(embedding_matrix)))
    m2.add(Bidirectional(LSTM(10, return_sequences=True, activation="relu")))
    m2.add(TimeDistributed(Dense(50, activation="relu")))
    m2.add(Dense(tag_len, activation="softmax"))
    m2.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    m3 = Sequential()
    m3.add(Embedding(vocab_len, embed_len,
                    embeddings_initializer=Constant(embedding_matrix)))
    m3.add(Bidirectional(LSTM(20, return_sequences=True, activation="relu")))
    m3.add(TimeDistributed(Dense(50, activation="relu")))
    m3.add(Dense(tag_len, activation="softmax"))
    m3.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    m1.fit(v_tr_sent, v_tr_target, batch_size=100, epochs=2, verbose=1)
    m2.fit(v_tr_sent, v_tr_target, batch_size=100, epochs=2, verbose=1)
    m3.fit(v_tr_sent, v_tr_target, batch_size=100, epochs=2, verbose=1)
    te_predicted1 = tag_sentences(te_sent, m1, wordmap, reverse_tagmap, maxlen)
    te_predicted2 = tag_sentences(te_sent, m2, wordmap, reverse_tagmap, maxlen)
    te_predicted3 = tag_sentences(te_sent, m3, wordmap, reverse_tagmap, maxlen)
    evaluate_ie(te_sent, te_tag, te_predicted1, entity_types,"eval1.txt")
    evaluate_ie(te_sent, te_tag, te_predicted2, entity_types,"eval2.txt")
    evaluate_ie(te_sent, te_tag, te_predicted3, entity_types,"eval3.txt")
main()