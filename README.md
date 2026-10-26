# Evaluating LLMs

**Bachelor's thesis project** · B.Sc. in Computer Engineering · Università degli Studi Roma Tre (2024)

How good are open LLMs, really, at the tasks people use them for most? This project benchmarks open-weight models served through the [Groq](https://groq.com/) API on four standard benchmarks, covering multiple-choice reasoning, broad knowledge, code generation and extractive question answering. It also measures how much **retrieval-augmented generation (RAG)** helps on the QA task.

| Script | Benchmark | Task | Metric |
|---|---|---|---|
| `ARC.py` | [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc) (Challenge) | grade-school science reasoning, multiple choice | accuracy |
| `MMLUTest.py` | [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | multi-domain knowledge, up to 10 options | accuracy |
| `HumanEval.py` | [OpenAI HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) | Python code generation from docstrings | pass@1 (all unit tests pass) |
| `NORAG.py` | [SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2) | closed-book QA (question only) | word-match accuracy, ROUGE-1/2/L |
| `RAG.py` | SQuAD v2 | QA with retrieved context | word-match accuracy, ROUGE-1/2/L, retrieval similarity |

## How it works

- **Multiple choice (ARC, MMLU-Pro).** Each question and its options are formatted into a prompt that asks for a single letter. The answer is parsed from the reply and compared with the gold label.
- **Code generation (HumanEval).** The model completes each function. The code is extracted from the reply (fenced blocks or raw text) and syntax-checked with `ast`. It is then executed against the benchmark's own `check()` tests, with `assert` statements rewritten so that every test case is counted. A task counts as solved only if all of its tests pass.
- **Question answering (SQuAD v2).** Unanswerable questions are filtered out. An answer is counted as correct when its words and the reference's words are contained one in the other, after punctuation removal and lower-casing. ROUGE scores are computed with `rouge-score` (ROUGE-2 only on multi-word references).
- **RAG pipeline.** Unique SQuAD contexts are embedded with `all-MiniLM-L6-v2` (Sentence-Transformers) and indexed with **FAISS** (`IndexFlatL2`). The top-2 passages are retrieved for each question and prepended to the prompt. The script also reports the average question–context cosine similarity, separately for correct and wrong answers, to relate retrieval quality to answer quality.

Default models (changeable in each script): `llama3-8b-8192` (ARC, MMLU-Pro, closed-book QA), `llama3-70b-8192` (HumanEval), `gemma2-9b-it` (RAG). Groq has since retired some of these model IDs, so replace them with a currently available one before running.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=<your key>     # https://console.groq.com/keys
```

## Usage

Each script is self-contained: it downloads its dataset from the Hugging Face Hub, queries the model and prints per-item logs and the final scores.

```bash
python ARC.py
python MMLUTest.py
python HumanEval.py
python NORAG.py     # 500 SQuAD v2 questions, no context
python RAG.py       # 300 SQuAD v2 questions, retrieved context
```

> ⚠️ `HumanEval.py` executes model-generated code with `exec`. Run it in an isolated environment.

> For a controlled RAG vs. closed-book comparison, set the same model and the same number of questions in `RAG.py` and `NORAG.py`.

## Tech stack

Python · Groq API · pandas · Hugging Face `datasets` · Sentence-Transformers · FAISS · scikit-learn · rouge-score
