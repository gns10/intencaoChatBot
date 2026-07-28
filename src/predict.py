import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import warnings

warnings.filterwarnings("ignore")


tokenizer = pickle.load(open(r"C:\Users\gusta\Desktop\intecaoChatBot\models\tokenizer.pkl", "rb"))

label_encoder = pickle.load(open(r"C:\Users\gusta\Desktop\intecaoChatBot\models\label_encoder.pkl", "rb"))

model = load_model(r"C:\Users\gusta\Desktop\intecaoChatBot\models\feelings.keras")

texto = "i want to change my pin code"

seq = tokenizer.texts_to_sequences([texto])

seq = pad_sequences(seq)

pred = model.predict(seq)

classe = label_encoder.inverse_transform([pred.argmax()])

print(pred.argmax())
print(classe[0])