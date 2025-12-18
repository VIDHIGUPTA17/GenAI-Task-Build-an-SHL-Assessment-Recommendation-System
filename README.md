# SHL Assessment Recommendation System
**AI Intern Assignment - Semantic Assessment Recommender**

🎯 Overview
3-phase semantic recommendation system for SHL Individual Test Solutions:
1. **Phase A**: Scrape → Clean → Embed → FAISS index
2. **Phase B**: FastAPI recommendation API + Web frontend
3. **Phase C**: Offline evaluation (Mean Recall@10) + test predictions

**Live Demo**:
- [API](http://genai-task-build-an-shl-assessment.onrender.com/)
- [Frontend](https://genai-task-build-an-shl-assessment-recommendation-system-drgun.streamlit.app/)
- [Predictions CSV in github file name (phase_c_predictions.csv) ]



## 📋 API Endpoints

| Endpoint | Method | Parameters | Response |
|----------|--------|------------|----------|
| `/health` | GET | - | `{"status": "healthy"}` |
| `/recommend` | POST | `query` (str), `k` (1-10) | JSON with `recommended_assessments` |


**Example**:
```bash
curl "http://genai-task-build-an-shl-assessment.onrender.com/recommend?query=jacva&k=5"
```

## 🏗️ Architecture

```
Query → Embedding → FAISS Search → Duration Filter → JSON Response
    ↓
Frontend Table: URL | Duration | Test Type | Adaptive | Remote
```

**Key Features**:
- Semantic search via `all-MiniLM-L6-v2` embeddings
- Duration constraint parsing ("max 40 minutes")
- Robust metadata extraction (handles string/int duration formats)
- URL normalization for evaluation

## 📊 Evaluation Results


**Submission CSV**: `phase_c_predictions.csv` (9 test queries × 10 recs)


## 📦 Requirements
```txt
fastapi==0.104.1
uvicorn==0.24.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
streamlit==1.28.1  # if using Streamlit frontend
requests==2.31.0
pandas==2.1.3
```

## 🔧 Customization

### Improve Recall@10:
1. Try different embedding models: `paraphrase-MiniLM-L6-v2`
2. Enrich embedding text: `title + description + test_type`
3. Add LLM reranking on top-K results

### Duration Parsing Examples:
```
"max 40 minutes" → 40
"about an hour" → 60
"budget is 30 mins" → 30
```

## 🧪 Testing Your Setup

1. **API Health**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Sample Query**:
   ```bash
   curl "http://localhost:8000/recommend?query=\"Python SQL developer\"&k=3" | jq
   ```



## 🤝 Deployment
- **API**: Render
- **Frontend**: Streamlit Cloud
- **FAISS Index**: Pre-build and upload to deployment

***

**Author**: [Vidhi Gupta]
