from transformers import TFAutoModel
bert_model = TFAutoModel. from_pretrained("distilbert-base-uncased")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
from datasets import load_dataset
tr_dataset = load_dataset("rotten_tomatoes", split="train") #train data split
tr_dataset = tr_dataset[:1066]
te_dataset = load_dataset("rotten_tomatoes", split="test") #test data split
#tr_dataset = tr_dataset.shuffle(seed=0)
#te_dataset = te_dataset.shuffle(seed=0)
tokenized_train = tokenizer(tr_dataset["text"] ,max_length=67, truncation=True,padding=True, return_tensors="tf")
tokenized_test = tokenizer(te_dataset["text"] ,max_length=67, truncation=True,padding=True, return_tensors="tf")
from tensorflow.keras.utils import to_categorical
train_y = to_categorical(tr_dataset["label"])
test_y = to_categorical(te_dataset["label"])
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
bert_model.trainable = False
maxlen = 67
token_ids = Input(shape=(maxlen,), dtype=tf.int32, name="token_ids")
attention_masks = Input(shape=(maxlen,), dtype=tf.int32, name="attention_masks")
bert_output = bert_model(token_ids,attention_mask=attention_masks)
dense_layer = Dense(64,activation="relu")(bert_output[0][:,0])
output = Dense(2,activation="softmax")(dense_layer)
model = Model(inputs=[token_ids,attention_masks],outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]], train_y, batch_size=25, epochs=3, verbose = 0)
score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]], test_y, verbose=0)
print("Accuracy on test data:", score[1])

#task 2
a = model.predict([tokenized_test["input_ids"], tokenized_test["attention_mask"]])
arr_correct_test = [] #to store the indexes of the correctly predicted sentences
# arr_correct_test1 = [] #to store the "values" of the correctly predicted sentences
num_correct = 10
for i in range(0,1066):
	if(num_correct!=0):
		if(a[i][0]>a[i][1]):
			label_pred=0
			if(te_dataset["label"][i] == label_pred):
				arr_correct_test.append(i)
				#arr_correct_test1.append(label_pred)
				num_correct = num_correct-1
		elif(a[i][0]<a[i][1]):
			label_pred=1
			if(te_dataset["label"][i] == label_pred):
				arr_correct_test.append(i)
				#arr_correct_test1.append(label_pred)
				num_correct = num_correct-1
	else:
		break
arr_incorrect_test = [] #to store the indexes of the incorrectly predicted sentences
# arr_incorrect_test1 = [] #to store the "values" of the incorrectly predicted sentences
num_incorrect = 10
for i in range(0,1066):
	if(num_incorrect!=0):
		if(a[i][0]>a[i][1]):
			label_pred=0
			if(te_dataset["label"][i] != label_pred):
				arr_incorrect_test.append(i)
				#arr_incorrect_test1.append(label_pred)
				num_incorrect = num_incorrect-1
		elif(a[i][0]<a[i][1]):
			label_pred=1
			if(te_dataset["label"][i] != label_pred):
				arr_incorrect_test.append(i)
				#arr_incorrect_test1.append(label_pred)
				num_incorrect = num_incorrect-1
	else:
		break
#printing the correctly predicted sentences for analysis
for i in range(0,10):
	print(te_dataset["text"][arr_correct_test[i]]+"\n")
#printing the incorrect predicted sentences for analysis
for i in range(0,10):
	print(te_dataset["text"][arr_incorrect_test[i]]+"\n")



#task 3

import numpy as np
from math import sqrt
def cosine_similarity(a, b) :
	return np.dot(a,b)/(sqrt(np.dot(a,a))*sqrt(np.dot(b,b)) )


#example 1
t = tokenizer(["The car is speeding.", "The automobile is parked."], max_length=9, truncation=True, padding=True, return_tensors = "tf")
output = bert_model(t["input_ids"], attention_mask=t["attention_mask"])
similarity = cosine_similarity(output[0][0][1], output[0][1][1])
print("Similarity:", similarity)


#example 2
t = tokenizer(["The happy child is playing.", "The sad child is crying."], max_length=9, truncation=True, padding=True, return_tensors = "tf")
output = bert_model(t["input_ids"], attention_mask=t["attention_mask"])
similarity = cosine_similarity(output[0][0][2], output[0][1][2])
print("Similarity:", similarity)

#example 3
t = tokenizer(["A cat is sleeping.", "The cats are playing."], max_length=9, truncation=True, padding=True, return_tensors = "tf")
output = bert_model(t["input_ids"], attention_mask=t["attention_mask"])
similarity = cosine_similarity(output[0][0][1], output[0][1][1])
print("Similarity:", similarity)

#example 4
t = tokenizer(["The dog chased the ball.", "The police chased the robber."], max_length=9, truncation=True, padding=True, return_tensors = "tf")
output = bert_model(t["input_ids"], attention_mask=t["attention_mask"])
similarity = cosine_similarity(output[0][0][2], output[0][1][2])
print("Similarity:", similarity)

#example 5
t = tokenizer(["The actor is rehearsing lines.", "The actress is performing on stage."], max_length=9, truncation=True, padding=True, return_tensors = "tf")
output = bert_model(t["input_ids"], attention_mask=t["attention_mask"])
similarity = cosine_similarity(output[0][0][1], output[0][1][1])
print("Similarity:", similarity)
