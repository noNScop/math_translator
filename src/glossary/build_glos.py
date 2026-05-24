import json
import re
from pathlib import Path
import spacy

nlp = spacy.load("en_core_web_sm")


INPUT_FILES = [
    ".\data\processed\source1.jsonl",
    ".\data\processed\source2.jsonl",
    ".\data\processed\source3.jsonl",
    ".\data\processed\source4.jsonl",
    ".\data\processed\source5.jsonl",
]

OUTPUT_FILE = ".\data\processed\master_final_67.jsonl"



def make_bow(entry):
    text = " ".join([
        entry.get("english", ""),
        #entry.get("polish", ""),
        #entry.get("domain", "")
    ])
    
    doc = nlp(entry.get("english", "").lower())

    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop and len(token) > 2
    ]
    
    return sorted(set(tokens))



def make_key(entry):
    english = entry.get("english", "").strip().lower()
    #polish = entry.get("polish", "").strip().lower()
    #domain = entry.get("domain", "").strip().lower()
    
    return (english)



def load_all(files):
    all_entries = []
    
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_entries.append(json.loads(line))
    
    return all_entries


def deduplicate(entries):
    seen = {}
    
    for e in entries:
        key = make_key(e)
        
        # keep first occurrence (or you could merge fields if needed)
        if key not in seen:
            seen[key] = e
    
    return list(seen.values())


def process():
    entries = load_all(INPUT_FILES)
    entries = deduplicate(entries)
    
    final = []
    for e in entries:
        e["bag_of_words"] = make_bow(e)
        final.append(e)

    final.sort(key=lambda x: (
        #x.get("domain", "").lower(),
        x.get("english", "").lower()
    ))
    
    return final


def save(entries, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    result = process()
    save(result, OUTPUT_FILE)
    print(f"Done. Wrote {len(result)} entries to {OUTPUT_FILE}")