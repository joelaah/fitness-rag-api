# ⚡ PulseFit RAG API – Production Intelligence Engine

[![Live Web App](https://img.shields.io/badge/Live_Web_App-PulseFit_AI-0284c7?style=for-the-badge&logo=googlechrome&logoColor=white)](https://joelaah.github.io/fitness_app/)
[![Frontend Repo](https://img.shields.io/badge/Frontend-Flutter_Web_%26_Mobile-02569B?style=for-the-badge&logo=flutter&logoColor=white)](https://github.com/joelaah/fitness_app)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production_Ready-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qdrant Cloud](https://img.shields.io/badge/Qdrant-Vector_Database-DC2626?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Cohere](https://img.shields.io/badge/Cohere-Cross--Encoder_Rerank-3949AB?style=for-the-badge)](https://cohere.com)
[![Render](https://img.shields.io/badge/Deployment-Render_Cloud-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://fitness-rag-api.onrender.com)

High-performance, two-stage Retrieval-Augmented Generation (RAG) backend powering **PulseFit AI**. Built with FastAPI, Qdrant Cloud HNSW vector indexing, Google GenAI embeddings, and Cohere Cross-Encoder reranking.

- 🌐 **Live Web Application:** [https://joelaah.github.io/fitness_app/](https://joelaah.github.io/fitness_app/)
- 📱 **Frontend Codebase:** [https://github.com/joelaah/fitness_app](https://github.com/joelaah/fitness_app)
- 🚀 **Live API Endpoint:** `https://fitness-rag-api.onrender.com`

---

## 🏛️ Pipeline Architecture

1. **Vector Database**: Qdrant Cloud AWS cluster, `documents` collection, Cosine Distance, 768-dim HNSW graph index.
2. **Asymmetric Embedding**:
   - Ingestion: `task_type="RETRIEVAL_DOCUMENT"` with Google `gemini-embedding-2`.
   - Querying: `task_type="RETRIEVAL_QUERY"` with Google `gemini-embedding-2`.
3. **Chunking Strategy**: Semantic paragraph-aware chunking with 600-char target window and 60-char sliding overlap.
4. **Two-Stage Retrieval & Reranking**:
   - **Stage 1 (Recall)**: Top-25 candidates via Qdrant HNSW ANN search.
   - **Stage 2 (Precision)**: Cohere Cross-Encoder (`rerank-v3.5`) cross-attention scoring. Chunks exceeding relevance threshold (>0.65) forwarded to generation.
5. **Deterministic Arithmetic Guardrails**:
   - Volume calculation, set/rep totals, and push/pull split ratios computed deterministically in Python to prevent mathematical hallucinations.
6. **Structured Output**: Pydantic `RecommendationResponse` schema validation guaranteeing robust contract adherence.

---

## 🛠️ Local Setup

```bash
# 1. Clone repo
git clone https://github.com/joelaah/fitness-rag-api.git
cd fitness-rag-api

# 2. Setup virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Configure environment (.env)
GEMINI_API_KEY=your_gemini_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
COHERE_API_KEY=your_cohere_key

# 5. Run server
uvicorn api:app --reload --port 8000
```

---

## 👨‍💻 Author

**Joel Lalruatkima**  
- **GitHub:** [@joelaah](https://github.com/joelaah)  
- **Email:** [joelapachuau64@gmail.com](mailto:joelapachuau64@gmail.com)
