import json
import spacy

nlp = spacy.load("en_core_web_sm")

with open("glossary.json", "r", encoding="utf-8") as f:
    glossary = json.load(f)

def extract_terms(text):
    doc = nlp(text)

    found = {}

    for token in doc:
        lemma = token.lemma_.lower()

        if lemma in glossary:
            found[lemma] = glossary[lemma]

    return found

text = "Integers form rings and fields."

terms = extract_terms(text)

print(terms)

def build_prompt(text, terms):
    glossary_text = "\n".join(
        [f"{k} → {v}" for k, v in terms.items()]
    )

    prompt = f"""
Use the glossary consistently.

Glossary:
{glossary_text}

Translate:
{text}
"""

    return prompt