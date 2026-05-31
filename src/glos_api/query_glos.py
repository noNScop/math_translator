import json
from pathlib import Path

import spacy

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


def inject_single_word_terms(text):

    doc = nlp(text)

    result = text

    replacements = []

    for token in doc:

        token_lemma = token.lemma_.lower()

        for entry in GLOSSARY:

            # Determine whether the glossary entry is truly one word
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

    # Apply replacements from the end of the string backwards
    for start, end, tagged in reversed(replacements):

        result = result[:start] + tagged + result[end:]

    return result

def extract_relevant_phrases(text, threshold=0.5):

    text_lemmas = set(lemmatize(text))

    relevant = []

    for entry in GLOSSARY:

        english = entry["english"]

        # skip single-word terms
        if len(english.split()) <= 1:
            continue

        bag = entry.get("bag_of_words", [])

        glossary_lemmas = (
            set(word.lower() for word in bag)
            if bag
            else set(lemmatize(english))
        )

        if not glossary_lemmas:
            continue

        overlap = len(text_lemmas & glossary_lemmas)

        score = overlap / len(glossary_lemmas)

        if score >= threshold:
            relevant.append(
                f'- {english} → {entry["polish"]}'
            )

    return relevant



def translate_with_glossary(
    text,
    system_prompt,
    user_prompt_template,
    use_glossary=True
):

    processed_text = text
    glossary_block = ""

    if use_glossary:

        # inline XML for single words
        processed_text = inject_single_word_terms(
            processed_text
        )

        # phrase glossary block
        phrases = extract_relevant_phrases(
            processed_text
        )

        if phrases:

            glossary_block = (
                "\n\nRelevant glossary:\n"
                + "\n".join(phrases)
            )

    final_user_prompt = user_prompt_template.format(
        text=processed_text
    )

    final_user_prompt += glossary_block

    effective_system_prompt = system_prompt

    if use_glossary:
        effective_system_prompt += "\n\n" + GLOSSARY_INSTRUCTIONS

    #print("glossary instructionsssss",GLOSSARY_INSTRUCTIONS)
    #print("system prompt",effective_system_prompt)
    #print("user prompt",final_user_prompt)

    return call_llm(
        effective_system_prompt,
        final_user_prompt
    )