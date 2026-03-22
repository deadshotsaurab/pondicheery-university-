import json

seeds = {
    "layman": [
        "wife", "husband", "son", "daughter", "death", "land", "pay",
        "village", "money", "year", "family", "child", "birth", "age",
        "person", "said", "time", "life", "name", "date"
    ],
    "student": [
        "evidence", "section", "provision", "authority", "obligation",
        "liability", "constitution", "regulation", "prosecution", "defendant",
        "agreement", "statutory", "amendment", "enforcement", "proceedings",
        "allegation", "compensation", "declaration", "penalty", "offence"
    ],
    "professional": [
        "testamentary", "intestate", "alienation", "devolution", "executrix",
        "probate", "bequest", "subrogation", "estoppel", "mandamus",
        "certiorari", "adjudication", "indemnification", "interlocutory",
        "encumbrance", "conveyance", "jurisprudence", "sequestration",
        "promissory", "abetment"
    ]
}

json.dump(seeds, open('seed_words.json', 'w', encoding='utf-8'), indent=2)
print('seed_words.json updated successfully')
for k, v in seeds.items():
    print(f'  {k}: {len(v)} seeds')