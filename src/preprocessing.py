import json
import pickle
import pandas as pd
from src.preprocessing import tokenizeFunc
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = None
encoder = None
X_train, X_val, X_test = None, None, None
y_train, y_val, y_test = None, None, None

with open(r"C:\Users\gusta\Desktop\intecaoChatBot\clinc150\clinc150_uci\data_full.json",
          "r", encoding="utf-8") as f:
    dados = json.load(f)

df_train = pd.DataFrame(dados["train"], columns=["text", "intent"])
df_val = pd.DataFrame(dados["val"], columns=["text", "intent"])
df_test = pd.DataFrame(dados["test"], columns=["text", "intent"])

#send to appropriate folder (./processed)
df_train.to_csv("./dataset/processed/train.csv", index=False, sep=";")
df_val.to_csv("./dataset/processed/val.csv", index=False, sep=";")
df_test.to_csv("./dataset/processed/test.csv", index=False, sep=";")

df_train["text_treated"] = df_train["text"].apply(tokenizeFunc)
df_val["text_treated"] = df_val["text"].apply(tokenizeFunc)
df_test["text_treated"] = df_test["text"].apply(tokenizeFunc)

def preprocess_data(df_train, df_val, df_test):
    global tokenizer
    tokenizer = Tokenizer()  
    tokenizer.fit_on_texts(df_train["text_treated"])
    X_train = tokenizer.texts_to_sequences(df_train["text_treated"])
    X_val = tokenizer.texts_to_sequences(df_val["text_treated"])
    X_test = tokenizer.texts_to_sequences(df_test["text_treated"])


    with open(r".\models\tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    return X_train, X_val, X_test, tokenizer

def encode_labels(df_train, df_val, df_test):
    global encoder
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(df_train["intent"])
    y_val = encoder.transform(df_val["intent"])
    y_test = encoder.transform(df_test["intent"])

    with open(r".\models\label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    return y_train, y_val, y_test

X_train, X_val, X_test, tokenizer = preprocess_data(df_train, df_val, df_test)
y_train, y_val, y_test = encode_labels(df_train, df_val, df_test)

X_train = pad_sequences(
    X_train,
    maxlen=20,
    padding="post"
)

X_val = pad_sequences(
    X_val,
    maxlen=20,
    padding="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=20,
    padding="post"
)
