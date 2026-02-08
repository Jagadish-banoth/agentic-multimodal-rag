# Copilot Instructions for Agentic Multimodal RAG

## Project Overview
This codebase implements an agentic, multimodal Retrieval-Augmented Generation (RAG) system. It dynamically retrieves and reasons over text, image, audio, and video data, using an agentic planner to orchestrate retrieval, verification, and answer generation.

## Architecture & Key Components
- **Control Plane**: Decision-making and orchestration
  - `planner/agentic_planner.py`: Classifies user intent, selects modalities, controls retries
  - `verification/verifier.py`: Checks answer faithfulness, triggers replanning if confidence is low
- **Data Plane**: Executes retrieval and generation
  - `ingestion/`: Multimodal data preprocessing (text, image, audio, video)
  - `knowledge_store/`: Hybrid memory (dense, sparse, graph)
  - `retrieval/`: Query-time retrieval (dense, sparse, reranking)
  - `fusion/context_fusion.py`: Context selection and compression
  - `generation/grounded_llm.py`: Grounded answer generation
- **Evaluation**: `evaluation/` for metrics and RAGAS pipeline
- **UI**: `ui/app.py` for user interface
- **Config**: Centralized in `config/settings.yaml`

## Developer Workflows
- **Environment**: Activate with `./venv/Scripts/Activate.ps1` (Windows PowerShell)
- **Ingestion**: `python -m ingestion.text_ingest` (or other modality)
- **Embedding/Indexing**: `python -m scripts.embed_and_index`
- **Retrieval Demo**: `python -m scripts.retrieve_demo`
- **Chat**: `python chat.py` (grounded LLM chat)
- **Orchestrator**: `python -m orchestrator.execution_engine` (non-interactive)
- **Evaluation**: `python -m evaluation.eval_retrieval`
- **RAGAS Pipeline**: `python scripts/run_ragas_pipeline.py --input evaluation/ragas_dev.jsonl --output artifacts/predictions.jsonl --mock`

## Patterns & Conventions
- **Separation of Concerns**: Control plane (planning/verification) is decoupled from data plane (retrieval/generation)
- **Hybrid Retrieval**: Combines dense, sparse, and graph-based search; reranking is always applied
- **Closed Agentic Loop**: Planner → Retrieval → Fusion → Generation → Verifier; loop retries if confidence is low
- **Config-Driven**: System behavior is controlled via `config/settings.yaml`
- **Artifacts**: Indices and embeddings are stored in `data/index/`; processed data in `data/processed/`
- **Evaluation**: RAGAS evaluation is reproducible and CI-integrated (see `.github/workflows/ragas-eval.yml`)

## Integration Points
- **Local LLM**: Pull with `ollama pull llama3:8b` (see README)
- **FAISS**: Used for vector search in `knowledge_store/vector_store.py`
- **BM25/Keyword**: `knowledge_store/sparse_index.py`
- **Graph Reasoning**: Optional, in `knowledge_store/knowledge_graph.py`

## Examples
- To add a new modality, extend `ingestion/` and update the planner logic
- To change retrieval strategy, modify `retrieval/` and planner
- For evaluation, add queries to `evaluation/ragas_dev.jsonl`

---
For more, see [README.md](../README.md) and config in [config/settings.yaml](../config/settings.yaml).
