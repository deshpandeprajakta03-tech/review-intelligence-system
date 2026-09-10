import re

def clean_text(text):
    """
    Clean the input text by:
    - Lowercasing
    - Removing punctuation
    - Stripping extra spaces
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)     # normalize whitespace
    return text.strip()
