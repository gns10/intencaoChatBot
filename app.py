import src.predict as pred

while True:

    texto = input("Você: ")

    if texto.lower() == "sair":
        break

    resposta = pred.predict_intent(texto)

    print("\nBot:", resposta)