import pandas as pd
from groq import Groq
import re

# Configuro il client Groq
client = Groq(
    api_key="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"
)

# Provo a caricare il dataset MMLU
try:
    splits = {'test': 'data/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/TIGER-Lab/MMLU-Pro/" + splits["test"])
    print("Dataset caricato con successo.")
    print(df.head())
except Exception as e:
    print(f"Errore nel download o nella lettura del dataset: {e}")
    df = None

# Funzione per estrarre la risposta corretta
def extract_answer(model_output):
    """Estrae solo la lettera A-Z dalla risposta del modello."""
    print(f"Estrazione della risposta dal modello:\n{model_output}")
    match = re.search(r'[A-Z]', model_output)
    if match:
        return match.group(0)
    else:
        print("Nessuna lettera trovata.")
        return None

# Funzione di valutazione
def evaluate_model(df, model_name):
    correct = 0
    total = len(df)

    for index, row in df.iterrows():
        question = row['question']
        options = row['options']  # Qui si accede direttamente alla lista di opzioni
        correct_answer = row['answer']
        
        # Formattazione del prompt per il modello
        prompt = f"Question: {question}\nChoices:\n"
        for i, option in enumerate(options):
            prompt += f"{chr(65+i)}. {option}\n"
        prompt += "Answer with one letter only for the corresponding option:"

        print(f"\nInizio valutazione per la domanda {index+1}/{total}")
        print(f"Prompt:\n{prompt}")
        print(f"Risposta corretta: {correct_answer}")

        # Chiamata al client Groq per ottenere la risposta del modello
        try:
            print(f"Chiamata al modello '{model_name}' per la domanda {index+1}...")
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )
            model_output = chat_completion.choices[0].message.content
            print(f"Risposta del modello:\n{model_output}")

            # Estrai la risposta del modello
            model_answer = extract_answer(model_output)

            # Confronta la risposta del modello con quella corretta
            if model_answer == correct_answer:
                correct += 1
                print("Risposta corretta!")
            else:
                print("Risposta sbagliata.")

        except Exception as e:
            print(f"Errore durante l'esecuzione del modello per la domanda {index+1}: {e}")

    # Calcolo dell'accuratezza
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Accuracy del modello '{model_name}': {accuracy:.2f}%")

# Chiamo la funzione di valutazione se il dataset è stato caricato
if df is not None:
    model_name = "llama3-8b-8192"  # Specifico il nome del modello da usare
    evaluate_model(df, model_name)
else:
    print("Nessun dataset disponibile per la valutazione.")
