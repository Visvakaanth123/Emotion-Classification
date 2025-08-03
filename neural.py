import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer,BertModel
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input



dataset = pd.read_csv('dreams.csv')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bertmodel = BertModel.from_pretrained('bert-base-uncased')

def encode(text):
    input = tokenizer(text,return_tensors='pt',truncation=True,padding=True,max_length=512)
    with torch.no_grad():
        output = bertmodel(**input)
        hidden_state = output.last_hidden_state
        cls_embeddings = hidden_state[0,0,:]
    return cls_embeddings.numpy()

x = np.vstack([encode(t)for t in dataset['text']])
y = dataset['label'].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=0.2)

model = Sequential([
    Input(shape=(768,)),
    Dense(64,activation='relu'),
    Dense(32,activation='relu'),
    Dense(16,activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=20,verbose=1)

text = 'i was talking happily with my friends'

encoded_text = encode(text).reshape(1,-1)
prediction = model.predict(encoded_text)
index = np.argmax(prediction)
confidence = prediction[0][index]

predicted_label = encoder.inverse_transform([int(index)])[0]
print(f"Predicted Sentiment: {predicted_label} \nConfidence: {confidence:.2f}")