import re

def extract_metadata(text):
    metadata = {}

    text_lower = text.lower()

    if "travel" in text_lower:
        metadata["policy_type"] = "travel"
    elif "security" in text_lower:
        metadata["policy_type"] = "security"
    elif "learning" in text_lower:
        metadata["policy_type"] = "learning"
    else:
        metadata["policy_type"] = "hr"

    doc_id_match = re.search(r'Document ID:\s*(.*)', text)
    if doc_id_match:
        metadata["doc_id"] = doc_id_match.group(1)

    date_match = re.search(r'Effective Date:\s*(.*)', text)
    if date_match:
        metadata["effective_date"] = date_match.group(1)

    return metadata
