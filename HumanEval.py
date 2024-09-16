import pandas as pd
import re
import math
import ast
from groq import Groq

# Configuro il client Groq
client = Groq(
    api_key="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"
)

# Provo a caricare il dataset
try:
    dataset = pd.read_parquet("hf://datasets/openai/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
    print("Dataset caricato con successo.")
    print(dataset.head())
except Exception as e:
    print(f"Errore nel download o nella lettura del dataset: {e}")
    dataset = None

def extract_test_code(test_code_str):
    """Estrae il codice di test dalla stringa METADATA e def"""
    print(f"Estrazione del codice di test dalla stringa: {test_code_str[:100]}...")  # Stampa i primi 100 caratteri
    match = re.search(r'def\s+check\(candidate\):([\s\S]+)', test_code_str)
    if match:
        code = match.group(1)
        print(f"Codice di test estratto:\n{code.strip()}")
        return code.strip()
    print("Nessun codice di test trovato.")
    return None

def extract_code_from_output(output):
    """Estrae solo il codice Python dal risultato del modello."""
    print(f"Estrazione del codice dal risultato del modello:\n{output}")  # Stampa i primi 100 caratteri

    # Cerca il codice delimitato da tripli backtick con o senza specifica di linguaggio
    match_backtick = re.search(r'```(?:python|Python)?\s*([\s\S]*?)```', output)
    if match_backtick:
        print("Codice Python trovato con backtick nel risultato del modello.")
        return match_backtick.group(1).strip()

    # Cerca il codice delimitato da triple virgolette singole con o senza specifica di linguaggio
    match_single_quote = re.search(r"'''(?:python|Python)?\s*([\s\S]*?)'''", output)
    if match_single_quote:
        print("Codice Python trovato con virgolette singole nel risultato del modello.")
        return match_single_quote.group(1).strip()

    # Nessun codice Python trovato con i delimitatori specificati
    print("Nessun codice Python formattato trovato. Ritorno l'output completo.")
    return output.strip()

def format_test_code(test_code):
    """Formatta e indenta correttamente il codice di test."""
    print(f"Formattazione del codice di test:\n{test_code}")
    lines = test_code.splitlines()
    formatted_lines = []
    inside_test_code = False

    for line in lines:
        if line.strip().startswith('assert '):
            inside_test_code = True
        if inside_test_code:
            formatted_lines.append('    ' + line.strip())
    
    formatted_code = '\n'.join(formatted_lines)
    print(f"Codice di test formattato:\n{formatted_code}")
    return formatted_code

def replace_assert_with_custom(code_str):
    """Sostituisce le istruzioni assert con custom_assert e aggiunge le parentesi necessarie."""
    # Utilizza una regex per catturare le espressioni di assert e sostituirle con custom_assert
    return re.sub(r'assert\s+(.+)', r'custom_assert(\1)', code_str)


def validate_code_syntax(code_str):
    """Controlla se il codice è sintatticamente corretto"""
    print(f"Validazione della sintassi del codice:\n{code_str}")  # Stampa i primi 100 caratteri
    try:
        ast.parse(code_str)
        print("Codice sintatticamente corretto.")
        return True, None
    except SyntaxError as e:
        print(f"Errore di sintassi trovato: {e}")
        return False, str(e)

def evaluate_dataset(data, model_name):
    correct = 0
    total = len(data)

    for index, row in data.iterrows():
        task_id = row['task_id']
        input_text = row['prompt']
        test_code = row['test']
        entry_point = row['entry_point']

        print(f"\nInizio valutazione per il Task ID: {task_id}")
        print(f"Prompt:\n{input_text}")
        
        # Estraggo il codice di test dalla colonna 'test'
        test_code_str = extract_test_code(test_code)
        if test_code_str:
            test_code_str = format_test_code(test_code_str)
            test_code_str = f"def check(candidate):\n{test_code_str}"  # Definisci check per eseguire i test

            # Sostituisco assert con custom_assert
            test_code_str = replace_assert_with_custom(test_code_str)

        # Eseguo la chiamata al client Groq per ottenere la risposta del modello
        try:
            print(f"Chiamata al modello '{model_name}' per il Task ID {task_id}...")
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": input_text,
                    }
                ],
                model=model_name,
            )
            model_output = chat_completion.choices[0].message.content
            print(f"Risposta del modello:\n{model_output}") 

            # Estraggo solo il codice dal model_output
            code_only_output = extract_code_from_output(model_output)

            # Valido la sintassi del codice
            is_valid, syntax_error = validate_code_syntax(code_only_output)
            if not is_valid:
                print(f"Errore di sintassi nel codice generato per il Task ID {task_id}: {syntax_error}")
                continue

            # Stampo il risultato per il debug
            print(f"Task ID: {task_id}")
            print(f"Input: {input_text}")
            print(f"Output del modello (codice estratto):\n{code_only_output}")
            print(f"Codice di test estratto:\n{test_code_str}\n")

            if code_only_output and test_code_str:
                try:
                    # Creo un dizionario per l'ambiente di esecuzione
                    test_globals = {}
                    # Aggiungo l'importazione mancante se necessario
                    test_globals['List'] = list
                    test_globals['math'] = math  # Aggiungo math se necessario

                    # Definisco custom_assert e sostituisco assert PRIMA di eseguire il codice di test
                    passed_tests = 0
                    test_results = []

                    def custom_assert(condition, message=""):
                        nonlocal passed_tests
                        if condition:
                            passed_tests += 1
                        else:
                            test_results.append(message)
                        print(f"Assert Custom: {'Passato' if condition else 'Fallito'} - {message}")

                    # Aggiungo custom_assert all'ambiente di esecuzione
                    test_globals['custom_assert'] = custom_assert

                    exec(code_only_output, test_globals)

                    candidate_function = test_globals.get(entry_point)

                    if candidate_function:
                        print(f"Funzione candidata '{entry_point}' trovata: {candidate_function}")
                    else:
                        print(f"Funzione candidata '{entry_point}' non trovata.")
                        continue

                    exec(test_code_str, test_globals)
                    check_function = test_globals.get('check')
                    
                    if check_function:
                        print("Funzione 'check' trovata.")
                        
                        # Conto i test-case
                        total_tests = test_code_str.count("custom_assert")
                        passed_tests = 0
                        test_results = []

                        # Eseguo i test e catturo i risultati
                        try:
                            check_function(candidate_function)
                        except Exception as e:
                            test_results.append(f"Errore durante l'esecuzione: {e}")
                        
                        print(f"Risultato: {passed_tests}/{total_tests} test-case superati per il Task ID {task_id}")
                        if test_results:
                            print(f"Dettagli sui test falliti:\n{test_results}")

                        if passed_tests == total_tests:
                            correct += 1
                    else:
                        print(f"Funzione 'check' non trovata per il Task ID {task_id}")
                except Exception as e:
                    print(f"Errore nell'esecuzione del test per il Task ID {task_id}: {e}")
            else:
                print(f"Nessun codice di test trovato per il Task ID {task_id}")

        except Exception as e:
            print(f"Errore durante l'esecuzione del modello per il Task ID {task_id}: {e}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Accuracy del modello '{model_name}': {accuracy:.2f}%")

# Chiamo la funzione di valutazione se il dataset è stato caricato
if dataset is not None:
    model_name = "llama3-70b-8192"  # Specifico il nome del modello da usare
    evaluate_dataset(dataset, model_name)
else:
    print("Nessun dataset disponibile per la valutazione.")
