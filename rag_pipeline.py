from openai import OpenAI
from pinecone import Pinecone
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def embed(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [r.embedding for r in res.data]

def expand(query):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Generate 3 variations of: " + query}]
    )
    variations = res.choices[0].message.content.split("\n")
    return [query] + variations

def retrieve_candidates(query, top_k=5):
    queries = expand(query)
    vectors = embed(queries)

    results = []

    for vec in vectors:
        res = index.query(vector=vec, top_k=top_k, include_metadata=True)
        results.extend(res["matches"])

    seen = set()
    unique = []
    for r in results:
        text = r["metadata"]["text"]
        if text not in seen:
            seen.add(text)
            unique.append(text)

    return unique[:top_k]

def rerank(query, docs):
    scored = []
    for doc in docs:
        score = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Score relevance (0-10):\nQuery: " + query + "\nDoc: " + doc}]
        )
        try:
            val = float(score.choices[0].message.content.strip())
        except:
            val = 0
        scored.append((doc, val))

    return [doc for doc, _ in sorted(scored, key=lambda x: x[1], reverse=True)]

def generate_answer(query, contexts):
    context_text = "\n\n".join(contexts)
    prompt = "Answer strictly using context:\n" + context_text + "\nQuestion:" + query

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content

def ask(query):
    docs = retrieve_candidates(query)
    docs = rerank(query, docs)
    answer = generate_answer(query, docs[:3])
    return answer, docs[:3]
