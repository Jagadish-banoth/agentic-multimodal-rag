# 🚀 Quick Start - FAANG Agentic Planner

## **5-Minute Setup**

### **Step 1: Pull Model (1 min)**
```powershell
ollama pull llama3:8b
```

### **Step 2: Start Ollama Server (Keep Running)**
```powershell
# In a SEPARATE PowerShell window
ollama serve

# Output should show:
# Listening on 127.0.0.1:11434
```

### **Optional: Use OpenRouter for Generation (Nemotron) + Fallback to Ollama**

1) Create a `.env` file (see `.env.example`) and set your key:

```text
OPENROUTER_API_KEY=sk-or-v1-...
```

2) By default, generation is configured as `provider: "auto"` in `config/settings.yaml`:
- If `OPENROUTER_API_KEY` (or `OP_TOKEN`) is set, it will use OpenRouter model `nvidia/nemotron-3-nano-30b-a3b:free`.
- If OpenRouter is unavailable or the key is missing, it falls back to local Ollama `llama3:8b`.

### **Step 3: Test Planner (In First Window)**
```powershell
cd D:\agentic-multimodal-rag
.\venv\Scripts\Activate.ps1
python -m planner.llm_planner
```

### **Step 4: Expected Output**
```
Query 1: What is the capital of France?
  Intent: factual
  Complexity: simple
  Modalities: ['text']
  Confidence: 0.95 (high)

[5 queries tested successfully]

METRICS SUMMARY:
  total_plans: 5
  avg_planning_time_ms: 450.2
  avg_confidence: 0.92
```

---

## **That's It! You're Done! 🎉**

---

## **If Something Goes Wrong**

### **Problem 1: "Connection refused"**
```
Error: Failed to connect to Ollama
Solution: Make sure ollama serve is running in another window
```

### **Problem 2: "Model not found"**
```
Error: llama3:8b not found
Solution: Run: ollama pull llama3:8b
```

### **Problem 3: "Still using fallback"**
```
Error: LLM not available, using fallback...
Solution: Check that Ollama window shows "Listening on 127.0.0.1:11434"
```

---

## **Next: Integrate with Orchestrator**

Once planner is working:

```powershell
# Update orchestrator to use new planner
python scripts/migrate_to_llm_planner.py --mode migrate

# Test end-to-end
python -m orchestrator.execution_engine
```

---

## **Pro Tips**

✅ Keep Ollama window open while developing  
✅ Monitor Ollama logs for errors  
✅ Test with simple queries first  
✅ Check `docs/LLM_PLANNER_GUIDE.md` for advanced usage  

---

**All Set!** 🚀

Your FAANG-grade agentic planner is ready to use!
