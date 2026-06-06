from sentence_transformers import CrossEncoder

# Downloads ~25MB on first run, then cached
_model = None

def get_model():
    global _model
    if _model is None:
        print('Loading reranker model (first time only)...')
        _model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _model

def rerank(query, docs, top_n=5):
    """Rerank docs by relevance to query, return top_n."""
    model = get_model()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)
    scored = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in scored[:top_n]]