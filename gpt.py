from transformers import pipeline
generator = pipeline("text-generation",model="distilgpt2")
generator("The cat jumped over the dog and ", max_length=30, num_return_sequences=5)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
inputs = tokenizer("The cat jumped over the dog and", return_tensors="tf")
from transformers import TFAutoModelForCausalLM
model = TFAutoModelForCausalLM.from_pretrained("distilgpt2")
outputs = model.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)
from datasets import load_dataset
dataset = load_dataset("Hellisotherpeople/DebateSum")
train_dataset = dataset["train"]
train_set = []
for i in range(0,150000):
	l = train_dataset[0]
	train_set.append(l["Full-Document"])
tokenizer.pad_token = "[PAD]"
train_encodings = tokenizer(train_set,return_tensors="tf", max_length=128,padding="max_length",truncation=True)
from datasets import Dataset
train_dataset = Dataset.from_dict(train_encodings)
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False, return_tensors="tf")
tf_train_set = model.prepare_tf_dataset(train_dataset,shuffle=True, batch_size=16, collate_fn=data_collator)
from transformers import AdamWeightDecay
optimizer = AdamWeightDecay(learning_rate=2e-5,weight_decay_rate=0.01)
model.compile(optimizer=optimizer)
model.fit(x=tf_train_set, epochs=3)
model.save_pretrained("new_distillgpt2")
from transformers import TFAutoModelForCausalLM
model = TFAutoModelForCausalLM.from_pretrained("C:/Users/Neeraja Vudhanthi/new_distillgpt2")
model1 = TFAutoModelForCausalLM.from_pretrained("distilgpt2")

#task 1

#example1
inputs = tokenizer("The heated debate centered around environmental ", return_tensors="tf")
outputs = model.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)

inputs = tokenizer("The heated debate centered around environmental ", return_tensors="tf")
outputs = model1.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)

#example2
inputs = tokenizer("Candidates sparred over healthcare, ", return_tensors="tf")
outputs = model.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)

inputs = tokenizer("Candidates sparred over healthcare, ", return_tensors="tf")
outputs = model1.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)


#example3
inputs = tokenizer("Education reform emerged as a key topic, ", return_tensors="tf")
outputs = model.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)

inputs = tokenizer("Education reform emerged as a key topic, ", return_tensors="tf")
outputs = model1.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)

#example4
inputs = tokenizer("The debaters exchanged passionate arguments on immigration, ", return_tensors="tf")
outputs = model.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)

inputs = tokenizer("The debaters exchanged passionate arguments on immigration,", return_tensors="tf")
outputs = model1.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)


#example5
inputs = tokenizer("Economic inequality took center stage, ", return_tensors="tf")
outputs = model.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)

inputs = tokenizer("Economic inequality took center stage, ", return_tensors="tf")
outputs = model1.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)


#task2
#example1
inputs = tokenizer("North American countries have established", return_tensors="tf")
outputs = model.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)

inputs = tokenizer("South American countries have established", return_tensors="tf")
outputs = model1.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)

#example2
inputs = tokenizer("Trade agreements in North America have led to increased", return_tensors="tf")
outputs = model.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)

inputs = tokenizer("Trade agreements in South America have led to increased", return_tensors="tf")
outputs = model1.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)

#example3
inputs = tokenizer("North American economies thrive on", return_tensors="tf")
outputs = model.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)

inputs = tokenizer("South American economies thrive on", return_tensors="tf")
outputs = model1.generate(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
tokenizer.batch_decode(outputs)