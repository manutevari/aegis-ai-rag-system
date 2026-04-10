from rag_pipeline import evaluate

test_cases = [
    ("Can I expense a taxi?", "taxi"),
    ("What is maternity leave policy?", "leave"),
]

evaluate(test_cases)
