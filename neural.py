import numpy as np
import pandas as pd
import torch 
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer,BertModel
import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer
import requests
import gradio as gr
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("dreams.csv")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bertmodel = BertModel.from_pretrained('bert-base-uncased')

def encode(text):
    input = tokenizer(text,return_tensors='pt',truncation=True,padding=True,max_length=512)
    with torch.no_grad():
        output = bertmodel(**input)
        hidden_state = output.last_hidden_state
        cls_embeddings = hidden_state[0,0,:]
    return cls_embeddings.numpy()

x = np.vstack([encode(t) for t in dataset['text']])
y = dataset['label'].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=0.2)

model = Sequential([
    InputLayer(shape=(768,)),
    Dense(64,activation='relu'),
    Dense(32,activation='relu'),
    Dense(16,activation='softmax')]
)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=20,verbose=1)

def llama(message,label,history):
    url = 'http://localhost:11434/v1/chat/completions'
    prompt = f"""You are a professional dream analyzer who analyzes the users dream and describe as to why it is classified as said emotion heres the dream {message} and here is the predicted label
    {label}"""

    data = {
        'messages':[{'role':'user','content':prompt}],
        'model':'llama3.2'
    }
    response = requests.post(url,json=data)
    if response.status_code == 200:
        response_data = response.json()
        llm_response = response_data['choices'][0]['message']['content']
        return llm_response

def gui(message,history):
    encoded_message = encode(message).reshape(1,-1)
    prediction = model.predict(encoded_message)
    index = np.argmax(prediction)
    confidence = prediction[0][index]
    label = encoder.inverse_transform([int(index)])[0]
    explanation = llama(message,label,history)
    return f"Predicted Label: {label}\n Confidence: {confidence:.2f}\n\n {explanation}"

interface = gr.ChatInterface(fn=gui).launch()