from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from src.model import model
from src.preprocessing import X_train, y_train, X_val, y_val

# Defina um objeto ModelCheckpoint para usar os melhores pesos para este modelo
checkpointer = ModelCheckpoint(filepath="./models/weights.best.keras", verbose=0, save_best_only=True) # save best model
learning_rate = 0.001

# Monitor para interromper o modelo antecipadamente quando a melhoria da perda de validação for mínima
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto', restore_best_weights=True) #Aqui utilizando early stopping

# Compilando o modelo e aplicação a função de custo Adam (utilizando a learning rate que configuramos anteriormente)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    validation_split=0.25,
                    verbose=1,callbacks=[monitor, checkpointer], epochs=45, batch_size=50, shuffle=True
)

# Salve os dados do modelo em um arquivo h5 
model.save('./models/feelings.keras')