import pandas as pd
import re
from groq import Groq

# Configuro il client Groq
client = Groq(
    api_key="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"
)

# Provo a caricare il dataset
def load_dataset():
    try:
        splits = {'train': 'ARC-Challenge/train-00000-of-00001.parquet', 'test': 'ARC-Challenge/test-00000-of-00001.parquet', 'validation': 'ARC-Challenge/validation-00000-of-00001.parquet'}
        df = pd.read_parquet("hf://datasets/allenai/ai2_arc/" + splits["train"])
        print("Dataset caricato con successo.")
        print(df.head())
        return df
    except Exception as e:
        print(f"Errore nel download o nella lettura del dataset: {e}")
        return None

# Funzione per creare il prompt da inviare al modello
def create_prompt(question, choices):
    """Genera un prompt formattato da inviare al modello."""
    choices_text = "\n".join([f"{label}: {text}" for label, text in zip(choices['label'], choices['text'])])
    prompt = f"Questiion: {question}\nChoices:\n{choices_text}\nAnswer using only the corresponding letter or number."
    return prompt

# Funzione per valutare il modello
def evaluate_model(df, model_name):
    correct = 0
    total = len(df)

    for index, row in df.iterrows():
        question = row['question']
        choices = row['choices']
        correct_answer = row['answerKey']
        task_id = row['id']

        # Creazione del prompt da inviare al modello
        prompt = create_prompt(question, choices)
        print(f"\nValutazione per il Task ID: {task_id}")
        print(f"Prompt:\n{prompt}")

        try:
            # Chiamata al modello
            print(f"Chiamata al modello '{model_name}' per il Task ID {task_id}...")
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )
            model_output = chat_completion.choices[0].message.content.strip()
            print(f"Risposta del modello: {model_output}")

            # Confronto della risposta del modello con la risposta corretta
            if model_output == correct_answer:
                correct += 1
                print(f"Risposta corretta! ({model_output})")
            else:
                print(f"Risposta sbagliata. ({model_output}), Risposta corretta: {correct_answer}")

        except Exception as e:
            print(f"Errore durante l'esecuzione del modello per il Task ID {task_id}: {e}")

    # Calcolo dell'accuratezza
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Accuracy del modello '{model_name}': {accuracy:.2f}%")

# Chiamo la funzione di valutazione se il dataset è stato caricato
dataset = load_dataset()
if dataset is not None:
    model_name = "llama3-8b-8192"  # Specifico il nome del modello da usare
    evaluate_model(dataset, model_name)
else:
    print("Nessun dataset disponibile per la valutazione.")
