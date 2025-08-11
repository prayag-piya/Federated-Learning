import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import keras

# Load tokenizer from pickle
with open(os.path.join("model", "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

# Load your TensorFlow model
model = keras.models.load_model("model/final_model.h5")  # adjust path accordingly

# Load config
with open(os.path.join("configs", "config.json"), "r") as f:
    config = json.load(f)

MAX_LEN = config.get("max_length", 20)

app = FastAPI()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    next_word: str

def predict_next(model, tokenizer, seed_text, max_len=20):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len, padding='pre')
    predicted_logits = model(token_list, training=False)
    predicted_id = tf.argmax(predicted_logits, axis=-1).numpy()[0]
    return tokenizer.index_word.get(predicted_id, '<UNK>')

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    next_word = predict_next(model, tokenizer, req.text, MAX_LEN)
    return PredictResponse(next_word=next_word) 