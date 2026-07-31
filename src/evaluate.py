import pandas as pd
from keras.models import load_model
from src.train import history
from src.preprocessing import X_test, y_test, encoder


model = load_model(r"C:\Users\gusta\Desktop\intecaoChatBot\models\feelings.keras")

loss, acc = model.evaluate(
    X_test,
    y_test
)

print(acc)
loss, acc = model.evaluate(X_test, y_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")
import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.legend()
plt.show()
print(history.history["accuracy"])
print(history.history["val_accuracy"])
from sklearn.metrics import classification_report
import numpy as np

y_pred = model.predict(X_test)
y_true = y_test
y_pred = np.argmax(model.predict(X_test), axis=1)

report = classification_report(y_true, y_pred, target_names=encoder.classes_,output_dict=True, 
    digits=3)

print(report)
df_report = pd.DataFrame(report).transpose()

df_report
df_report.sort_values("f1-score", inplace=True)
df_report.to_excel("./classification_report.xlsx", index=True)