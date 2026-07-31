from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from src.preprocessing import tokenizer

model = Sequential([
    Embedding(
        input_dim=len(tokenizer.word_index)+1,
        output_dim=128,
        input_length=30
    ),

    Bidirectional(
        LSTM(128)
    ),
    Dropout(0.2),
    
    Dense(150, activation="softmax")
])