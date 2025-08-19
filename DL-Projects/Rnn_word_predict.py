import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from sklearn.preprocessing import LabelBinarizer

text="hello world. this is an RNN demo for learning purpose."*10
text

char=list(set(text))
idex_to_char={i:j for i,j in enumerate(char)}
char_to_index={j:i for i,j in enumerate(char)}
vocab_size=len(char)


seq_len=10
x_data=[]
y_data=[]
for i in range(len(text)-seq_len):
  input=text[i:i+seq_len]
  idex_seq=text[i+seq_len]
  x_data.append([char_to_index[c] for c in input])
  y_data.append([char_to_index[idex_seq]])

x=np.array(x_data)
y=tf.keras.utils.to_categorical(y_data)

model=Sequential([
    Embedding(input_dim=vocab_size,output_dim=10,input_length=seq_len),SimpleRNN(288),
    Dense(vocab_size,activation="softmax")

])

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(x,y,epochs=10)

def predicted(seed_text,num_vars):
  input=[char_to_index[char] for char in seed_text[-seq_len:]]
  input_shape=np.reshape(input,(1,seq_len))
  prediction=model.predict(input_shape)
  index=np.argmax(prediction)
  seed_text=seed_text+idex_to_char[index]
  return seed_text

print(predicted("hello world",100))