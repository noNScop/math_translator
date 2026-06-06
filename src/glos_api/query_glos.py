import json
from pathlib import Path

import spacy
from collections import defaultdict

from api import call_llm
from prompts import GLOSSARY_INSTRUCTIONS
from config import CONFIG


GLOSSARY_FILE = Path(
    CONFIG["paths"]["glossary"]
)

nlp = spacy.load("en_core_web_sm")


def load_glossary():
    glossary = []

    with open(GLOSSARY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            glossary.append(json.loads(line))

    return glossary


def lemmatize(text):
    doc = nlp(text)

    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_space
    ]


GLOSSARY = load_glossary()


PHRASE_GLOSSARY = []

for entry in GLOSSARY:

    english = entry["english"]

    bag = entry.get("bag_of_words", [])

    if not bag:
        continue

    PHRASE_GLOSSARY.append(
        {
            "english": english,
            "polish": entry["polish"],
            "lemmas": set(word.lower() for word in bag),
        }
    )


INVERTED_INDEX = defaultdict(set)

for i, entry in enumerate(PHRASE_GLOSSARY):
    for lemma in entry["lemmas"]:
        INVERTED_INDEX[lemma].add(i)


def inject_single_word_terms(text):

    doc = nlp(text)

    result = text

    replacements = []

    for token in doc:

        token_lemma = token.lemma_.lower()

        for entry in GLOSSARY:

            english_lemmas = lemmatize(entry["english"])

            if len(english_lemmas) != 1:
                continue

            glossary_lemma = english_lemmas[0]

            if token_lemma == glossary_lemma:

                tagged = (
                    f'<term target="{entry["polish"]}">'
                    f'{token.text}'
                    f'</term>'
                )

                replacements.append(
                    (
                        token.idx,
                        token.idx + len(token.text),
                        tagged
                    )
                )

    for start, end, tagged in reversed(replacements):

        result = result[:start] + tagged + result[end:]

    return result


def extract_relevant_phrases(
    text,
    threshold=0.5
):

    text_lemmas = set(lemmatize(text))

  
    candidate_ids = set()

    for lemma in text_lemmas:
        candidate_ids.update(INVERTED_INDEX.get(lemma, set()))

    relevant = []

    for idx in candidate_ids:

        entry = PHRASE_GLOSSARY[idx]

        glossary_lemmas = entry["lemmas"]

        overlap = len(text_lemmas & glossary_lemmas)

        score = overlap / len(glossary_lemmas)

        if score >= threshold:

            relevant.append(
                f'- {entry["english"]} → {entry["polish"]}'
            )

    return relevant


def translate_with_glossary(
    text,
    system_prompt,
    user_prompt_template,
    use_glossary=True,
    llm_fn=None,
):

    processed_text = text
    glossary_block = ""

    if use_glossary:

        phrases = extract_relevant_phrases(
            processed_text,
            threshold=CONFIG["glossary"]["phrase_match_threshold"]
        )

        if phrases:
            glossary_block = (
                "\n\nRelevant glossary:\n"
                + "\n".join(phrases)
            )

    final_user_prompt = (
        user_prompt_template.format(
            text=processed_text
        )
    )

    final_user_prompt += glossary_block
    fn = llm_fn if llm_fn is not None else call_llm
    return fn(
        system_prompt,
        final_user_prompt
    )