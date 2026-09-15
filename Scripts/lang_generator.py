metadata = {
    # Nouns
    "obama": {"meaning": "man", "category": "noun"},
    "guguGaga": {"meaning": "child", "category": "noun"},
    "osama": {"meaning": "woman", "category": "noun"},
    "kuku": {"meaning": "dog", "category": "noun"},
    "neko": {"meaning": "cat", "category": "noun"},
    "dagio": {"meaning": "t-rex", "category": "noun"},
    "madre": {"meaning": "mother", "category": "noun"},
    "padre": {"meaning": "father", "category": "noun"},
    "dumdum": {"meaning": "students", "category": "noun"},
    "kaka": {"meaning": "crocodile", "category": "noun"},
    "chichi": {"meaning": "sparrow", "category": "noun"},
    "kawkaw": {"meaning": "crow", "category": "noun"},
    "resta": {"meaning": "wolf", "category": "noun"},
    "femi": {"meaning": "mountain", "category": "noun"},
    "vomi": {"meaning": "chair", "category": "noun"},
    "luma": {"meaning": "star", "category": "noun"},
    "pyro": {"meaning": "fire", "category": "noun"},
    "ghar": {"meaning": "house", "category": "noun"},
    "damar": {"meaning": "road", "category": "noun"},
    "kafka": {"meaning": "book", "category": "noun"},

    # Verbs
    "bhago": {"meaning": "run", "category": "verb"},
    "yumyum": {"meaning": "eat", "category": "verb"},
    "nini": {"meaning": "sleep", "category": "verb"},
    "toing": {"meaning": "jump", "category": "verb"},
    "snan": {"meaning": "swim", "category": "verb"},
    "icarus": {"meaning": "fly", "category": "verb"},
    "jojo": {"meaning": "walk", "category": "verb"},
    "perv": {"meaning": "look", "category": "verb"},
    "sisy": {"meaning": "carry", "category": "verb"},
    "bobder": {"meaning": "build", "category": "verb"},
    "fenu": {"meaning": "open", "category": "verb"},
    "zari": {"meaning": "close", "category": "verb"},
    "loti": {"meaning": "write", "category": "verb"},
    "rime": {"meaning": "read", "category": "verb"},
    "paku": {"meaning": "push", "category": "verb"},
    "soro": {"meaning": "pull", "category": "verb"},
    "davi": {"meaning": "climb", "category": "verb"},
    "nexo": {"meaning": "fall", "category": "verb"},
    "naku": {"meaning": "sing", "category": "verb"},
    "savi": {"meaning": "dance", "category": "verb"},

    # Adjectives
    "fira": {"meaning": "red", "category": "adj"},
    "boku": {"meaning": "blue", "category": "adj"},
    "sena": {"meaning": "green", "category": "adj"},
    "moro": {"meaning": "yellow", "category": "adj"},
    "leni": {"meaning": "big", "category": "adj"},
    "vaku": {"meaning": "small", "category": "adj"},
    "desa": {"meaning": "fast", "category": "adj"},
    "roni": {"meaning": "slow", "category": "adj"},
    "kema": {"meaning": "strong", "category": "adj"},
    "zeli": {"meaning": "weak", "category": "adj"},
    "pira": {"meaning": "hot", "category": "adj"},
    "nemo": {"meaning": "cold", "category": "adj"},
    "jaku": {"meaning": "bright", "category": "adj"},
    "henu": {"meaning": "dark", "category": "adj"},
    "bora": {"meaning": "new", "category": "adj"},
    "tenu": {"meaning": "old", "category": "adj"},
    "fomi": {"meaning": "happy", "category": "adj"},
    "yera": {"meaning": "sad", "category": "adj"},
    "dani": {"meaning": "loud", "category": "adj"},
    "loku": {"meaning": "quiet", "category": "adj"},
}

nouns = [
    "obama",
    "guguGaga",
    "osama",
    "kuku",
    "neko",
    "dagio",
    "madre",
    "padre",
    "dumdum",
    "kaka",
    "chichi",
    "kawkaw",
    "resta",
    "femi",
    "vomi",
    "luma",
    "pyro",
    "ghar",
    "damar",
    "kafka"
]

verbs = [
    "bhago",
    "yumyum",
    "nini",
    "toing",
    "snan",
    "icarus",
    "jojo",
    "perv",
    "sisy",
    "bobder",
    "fenu",
    "zari",
    "loti",
    "rime",
    "paku",
    "soro",
    "davi",
    "nexo",
    "naku",
    "savi"
]

adjectives = [
    "fira",
    "boku",
    "sena",
    "moro",
    "leni",
    "vaku",
    "desa",
    "roni",
    "kema",
    "zeli",
    "pira",
    "nemo",
    "jaku",
    "henu",
    "bora",
    "tenu",
    "fomi",
    "yera",
    "dani",
    "loku"
]



sentences = []
for adj in adjectives:
    for noun in nouns:
        for verb in verbs:
            sentences.append(
                f"{adj} {noun} {verb}"
            )

with open("Data/langv1.txt", 'w') as f:
    f.write('\n'.join(sentences))

# print(sentences[:100])