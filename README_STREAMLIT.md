# Intent Chatbot — Streamlit

## Executar

No diretório do projeto:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fluxo

Usuário → app.py → predict.py → modelo → intenção → response_service.py → intents.json

O modelo, tokenizer e label encoder são carregados uma única vez por processo.

## Respostas

Preencha `data/intents.json` com as 150 intenções e suas respostas/links.

Formato:

```json
{
    "balance": {
        "response": "Seu saldo está disponível no aplicativo.",
        "action": null,
        "url": "https://exemplo.com"
    }
}
```
