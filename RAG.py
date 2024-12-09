import pandas as pd
import numpy as np
import time
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
def filter_non_empty_answers(df, max_rows=300):
    filtered_rows = []
    for example in df:
        if len(example['answers']['text']) > 0:  # Verifica che ci sia una risposta
            filtered_rows.append(example)
        if len(filtered_rows) == max_rows:  # Ferma quando raggiunge esattamente 100 righe non vuote
            break
    if len(filtered_rows) < max_rows:
        raise ValueError(f"Non ci sono abbastanza righe non vuote. Trovate solo {len(filtered_rows)} righe.")
    return filtered_rows

# Estrae i contesti unici dal dataset
def extract_unique_contexts(df, num_contexts=300):
    # Estrai i contesti unici per evitare duplicati
    contexts = [example['context'] for example in df]
    unique_contexts = list(pd.Series(contexts).drop_duplicates().head(num_contexts))
    return unique_contexts

# Funzione per generare il prompt da inviare al modello
def create_prompt(question, context):
    prompt = f"Question: {question}\nContext: {context}\nAnswer with a few words."
    return prompt

# Configura l'indice FAISS e indicizza i documenti recuperati (context)
def setup_faiss_index(embedding_model, contexts):
    embeddings = embedding_model.encode(contexts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"Numero di documenti indicizzati: {index.ntotal}")
    return index

# Recupera i documenti correlati usando FAISS
def retrieve_documents(question, index, embedding_model, contexts, k=2):
    question_embedding = embedding_model.encode([question]).astype('float32')
    distances, indices = index.search(question_embedding, k)
    retrieved_docs = [contexts[idx] for idx in indices[0]]
    retrieved_embeddings = embedding_model.encode(retrieved_docs).astype('float32')
    similarity_scores = cosine_similarity([question_embedding[0]], retrieved_embeddings).flatten()
    avg_similarity = np.mean(similarity_scores)
    return "\n".join(retrieved_docs), indices[0], avg_similarity, len(retrieved_docs)

# Genera la risposta usando il modello Groq
def generate_answer(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it",  
            max_tokens=15  
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

# Valuta il modello usando retrieval e Groq per il question-answering
def evaluate_model(df, contexts, index, embedding_model):
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

        # Recupera i documenti correlati (contesti)
        retrieved_context, retrieved_indices, avg_similarity, retrieved_count = retrieve_documents(question, index, embedding_model, contexts)
        print(f"Iterazione {num + 1} - Similarità tra input e documenti retrieved: {avg_similarity:.2f}")
        print(f"Iterazione {num + 1} - Numero di documenti retrieved: {retrieved_count}")

        prompt = create_prompt(question, retrieved_context)

        # Usa il modello per generare la risposta
        model_output = generate_answer(f"{prompt}")
        print(f"Iterazione {num + 1} - Risposta generata: {model_output}")
        print(f"Iterazione {num + 1} - Risposta attesa: {answer}")

        if check_if_all_words_present(answer, model_output):
            total_correct += 1
            total_similarity_correct += avg_similarity
            print(f"Iterazione {num + 1} - Risposta corretta! ({model_output})")
            print(f"Siamo a {total_correct} risposte corrette")
        else:
            total_incorrect += 1
            total_similarity_incorrect += avg_similarity
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
        avg_similarity_correct = total_similarity_correct / total_correct
        print(f"Similarità media per risposte corrette: {avg_similarity_correct:.2f}")
    if total_incorrect > 0:
        avg_similarity_incorrect = total_similarity_incorrect / total_incorrect
        print(f"Similarità media per risposte errate: {avg_similarity_incorrect:.2f}")

    # Calcolo delle medie
    avg_rouge1 = total_rouge1 / len(df)
    avg_rougeL = total_rougeL / len(df)
    avg_rouge2 = total_rouge2 / valid_rows_for_rouge2 if valid_rows_for_rouge2 > 0 else 0

    print(f"ROUGE-1 medio: {avg_rouge1:.2f}, ROUGE-2 medio (solo risposte > 1 parola): {avg_rouge2:.2f}, ROUGE-L medio: {avg_rougeL:.2f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tempo totale per esaminare l'intero dataset: {elapsed_time:.2f} secondi")
    print(f"Numero totale di risposte corrette: {total_correct}")

# Esecuzione del codice
dataset = load_dataset()
if dataset is not None:
    filtered_dataset = filter_non_empty_answers(dataset, 300)  # Assicura che ci siano esattamente 100 righe non vuote
    contexts = extract_unique_contexts(filtered_dataset)
    print(f"Contesti unici estratti: {contexts[:5]}")

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_index = setup_faiss_index(embedding_model, contexts)

    evaluate_model(filtered_dataset, contexts, faiss_index, embedding_model)
else:
    print("Nessun dataset disponibile per la valutazione.")
