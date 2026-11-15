from sentence_transformers import SentenceTransformer, util

_model = None


def rerank(claim, docs, top=3):
    """Keep the `top` passages most similar to the claim (cosine similarity)."""
    global _model
    if not docs:
        return []
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    c = _model.encode(claim, convert_to_tensor=True)
    d = _model.encode([f"{x['title']}. {x['text']}" for x in docs], convert_to_tensor=True)
    scores = util.cos_sim(c, d)[0].tolist()
    ranked = sorted(zip(scores, docs), key=lambda p: -p[0])
    return [dict(doc, score=round(s, 3)) for s, doc in ranked[:top]]
