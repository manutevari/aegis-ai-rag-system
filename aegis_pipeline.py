# ============================
# METADATA EXTRACTION (FINAL)
# ============================

def extract_metadata(text):
    metadata = {
        "document_id": "DOC-001",
        "category": "General",
        "owner": "Unknown",
        "effective_date": "Unknown",
        "headers": []
    }

    lines = text.split("\n")

    for line in lines:
        l = line.lower()

        if "policy category" in l:
            metadata["category"] = line.split(":")[-1].strip()

        if "policy owner" in l:
            metadata["owner"] = line.split(":")[-1].strip()

        if "effective date" in l:
            metadata["effective_date"] = line.split(":")[-1].strip()

        # Capture headers
        if re.match(r"#{1,3} ", line):
            metadata["headers"].append(line.strip())

    return metadata
