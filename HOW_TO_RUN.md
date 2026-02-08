# How to Run End-to-End

## Prerequisites

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Ensure Ollama is running (in separate terminal)
ollama serve
```

## Step-by-Step Commands

### Step 1: Ingest Data
```powershell
# Process documents from data/raw/text/
python -m ingestion.text_ingest
```

### Step 2: Build Index
```powershell
# Generate embeddings and build FAISS + BM25 index
python -m scripts.embed_and_index
```

### Step 3: Test Retrieval (Optional)
```powershell
python -m scripts.retrieve_demo
```

### Step 4: Run Full Pipeline

**Option A: Single Query (Orchestrator)**
```powershell
python -m orchestrator.execution_engine
```

**Option B: Interactive Chat**
```powershell
python chat.py
```

**Option C: RAGAS Evaluation**
```powershell
python scripts\run_ragas_pipeline.py --input evaluation\ragas_dev.jsonl --output artifacts\predictions.jsonl
```

## Quick Reference Table

| Step | Command | Purpose |
|------|---------|---------|
| Activate | `.\venv\Scripts\Activate.ps1` | Load environment |
| Ingest | `python -m ingestion.text_ingest` | Process documents |
| Index | `python -m scripts.embed_and_index` | Build FAISS + BM25 |
| Retrieve | `python -m scripts.retrieve_demo` | Test retrieval |
| Chat | `python chat.py` | Interactive Q&A |
| Evaluate | `python scripts\run_ragas_pipeline.py ...` | Run RAGAS eval |

## Configuration

- **Config file**: `config/settings.yaml`
- **API Key**: Set `OPENROUTER_API_KEY` in `.env` file
- **Provider**: Auto-detects OpenRouter or falls back to Ollama

## Expected Pipeline Flow

```
Query → Planner → Retrieval → Fusion → Generation → Verification → Response
```

- **Planner**: Classifies intent and selects modalities
- **Retrieval**: Dense (FAISS) + Sparse (BM25) + Reranking
- **Fusion**: Selects top chunks within token budget
- **Generation**: OpenRouter/Ollama LLM
- **Verification**: NLI faithfulness check
