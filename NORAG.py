import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from groq import Groq
from rouge_score import rouge_scorer
import string

# Configuro il client Groq con la tua API key
client = Groq(api_key="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")

# Carica il dataset SQuAD v2
def load_dataset():
    try:
        from datasets import load_dataset
        dataset = load_dataset('rajpurkar/squad_v2')
        print("Dataset caricato con successo.")
        return dataset['validation']  # Usa 'validation' per SQuAD v2
    except Exception as e:
        print(f"Errore nel download o nella lettura del dataset: {e}")
        return None

# Filtra esattamente 100 righe non vuote
def filter_non_empty_answers(df, max_rows=500):
    filtered_rows = []
    for example in df:
        if len(example['answers']['text']) > 0:  # Verifica che ci sia una risposta
            filtered_rows.append(example)
        if len(filtered_rows) == max_rows:  # Ferma quando raggiunge esattamente 100 righe non vuote
            break
    if len(filtered_rows) < max_rows:
        raise ValueError(f"Non ci sono abbastanza righe non vuote. Trovate solo {len(filtered_rows)} righe.")
    return filtered_rows

# Funzione per generare il prompt da inviare al modello
def create_prompt(question):
    prompt = f"Question: {question}\nAnswer with a few words."
    return prompt

# Genera la risposta usando il modello Groq
def generate_answer(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",  
            max_tokens=10  
        )
        generated_text = chat_completion.choices[0].message.content.strip()
        return generated_text
    except Exception as e:
        print(f"Errore durante la generazione della risposta: {e}")
        return ""

# Funzione per rimuovere punteggiatura
def clean_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower()

# Funzione per verificare se la risposta generata contiene tutte le parole della risposta corretta
def check_if_all_words_present(correct_answer, generated_answer):
    correct_words = set(clean_text(correct_answer).split())
    generated_words = set(clean_text(generated_answer).split())
    return correct_words.issubset(generated_words) or generated_words.issubset(correct_words)

# Funzione per calcolare ROUGE score
def calculate_rouge_score(generated_answer, correct_answer):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(correct_answer, generated_answer)
    return scores

# Valuta il modello senza retrieval (RAG) per il question-answering
def evaluate_model(df, embedding_model):
    total_similarity_correct = 0  
    total_similarity_incorrect = 0  
    total_correct = 0  
    total_incorrect = 0  
    total_rouge1 = 0  
    total_rouge2 = 0  
    total_rougeL = 0  
    valid_rows_for_rouge2 = 0  # Conta solo le righe valide per ROUGE-2

    start_time = time.time()  

    for num, row in enumerate(df):  
        question = row['question']
        answer = row['answers']['text'][0]  

        # Genera il prompt con solo la domanda
        prompt = create_prompt(question)

        # Usa il modello per generare la risposta
        model_output = generate_answer(f"{prompt}")
        print(f"Iterazione {num + 1} - Risposta generata: {model_output}")
        print(f"Iterazione {num + 1} - Risposta attesa: {answer}")

        if check_if_all_words_present(answer, model_output):
            total_correct += 1
            print(f"Iterazione {num + 1} - Risposta corretta! ({model_output})")
            print(f"Siamo a {total_correct} risposte corrette")
        else:
            total_incorrect += 1
            print(f"Iterazione {num + 1} - Risposta errata! ({model_output}), Risposta corretta: {answer}")

        # Calcola e somma il ROUGE score
        rouge_scores = calculate_rouge_score(model_output, answer)
        print(f"Iterazione {num + 1} - ROUGE-1: {rouge_scores['rouge1'].fmeasure:.2f}, ROUGE-2: {rouge_scores['rouge2'].fmeasure:.2f}, ROUGE-L: {rouge_scores['rougeL'].fmeasure:.2f}")
        total_rouge1 += rouge_scores['rouge1'].fmeasure
        total_rougeL += rouge_scores['rougeL'].fmeasure

        # Conta solo per ROUGE-2 se la risposta ha più di una parola
        if len(answer.split()) > 1:
            valid_rows_for_rouge2 += 1
            total_rouge2 += rouge_scores['rouge2'].fmeasure

    if total_correct > 0:
        print(f"Numero totale di risposte corrette: {total_correct}")
    
    # Calcolo delle medie
    avg_rouge1 = total_rouge1 / len(df)
    avg_rougeL = total_rougeL / len(df)
    avg_rouge2 = total_rouge2 / valid_rows_for_rouge2 if valid_rows_for_rouge2 > 0 else 0

    print(f"ROUGE-1 medio: {avg_rouge1:.2f}, ROUGE-2 medio (solo risposte > 1 parola): {avg_rouge2:.2f}, ROUGE-L medio: {avg_rougeL:.2f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tempo totale per esaminare l'intero dataset: {elapsed_time:.2f} secondi")

# Esecuzione del codice
dataset = load_dataset()
if dataset is not None:
    filtered_dataset = filter_non_empty_answers(dataset, 500)  # Assicura che ci siano esattamente 100 righe non vuote
    print(f"Numero di righe filtrate: {len(filtered_dataset)}")

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    evaluate_model(filtered_dataset, embedding_model)
else:
    print("Nessun dataset disponibile per la valutazione.")
