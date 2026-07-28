import pickle
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import Tokenizer


def preprocess_data(df_train, df_val, df_test):
    tokenizer = Tokenizer()  
    tokenizer.fit_on_texts(df_train["text_treated"])
    X_train = tokenizer.texts_to_sequences(df_train["text_treated"])
    X_val = tokenizer.texts_to_sequences(df_val["text_treated"])
    X_test = tokenizer.texts_to_sequences(df_test["text_treated"])


    with open(r".\models\tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    return X_train, X_val, X_test, tokenizer

def encode_labels(df_train, df_val, df_test):
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(df_train["intent"])
    y_val = encoder.transform(df_val["intent"])
    y_test = encoder.transform(df_test["intent"])

    with open(r".\models\label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    return y_train, y_val, y_test
