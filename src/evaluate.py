import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras") # Ignora avis

model = load_model("./models/feelings.keras")
with open("./models/history.pkl", "rb") as f:
    history = pickle.load(f)

X_test = np.load("./models/X_test.npy")
y_test = np.load("./models/y_test.npy")

with open("./models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

loss, acc = model.evaluate(X_test, y_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")

plt.plot(history["accuracy"], label="train")
plt.plot(history["val_accuracy"], label="val")
plt.legend()
plt.show()
print(history["accuracy"])
print(history["val_accuracy"])


y_true = y_test
y_pred = np.argmax(model.predict(X_test), axis=1)

report = classification_report(y_true, y_pred, target_names=encoder.classes_,output_dict=True, 
    digits=3)

#add matriz de confusão
#add top 3 accuracy
df_report = pd.DataFrame(report).transpose()
df_report.sort_values("f1-score", inplace=True)
df_report.to_excel("./classification_report.xlsx", index=True)