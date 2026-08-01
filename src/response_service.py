from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
INTENTS_PATHS = [
    BASE_DIR / "data" / "intents.json",
    BASE_DIR / "intents.json",
]

DEFAULT_RESPONSE = (
    "Identifiquei a intenção, mas ainda não existe uma resposta cadastrada "
    "para ela no intents.json."
)


def load_intents():
    for path in INTENTS_PATHS:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
    return {}


def get_response(intent: str):
    intents = load_intents()
    item = intents.get(intent)

    if item is None:
        return {
            "response": DEFAULT_RESPONSE,
            "url": None,
            "action": None,
        }

    return {
        "response": item.get("response", DEFAULT_RESPONSE),
        "url": item.get("url"),
        "action": item.get("action"),
    }
