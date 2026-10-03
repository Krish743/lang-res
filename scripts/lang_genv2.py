import random


people = [
    "obama", "osama", "madre", "padre", "dumdum"
]

animals = [
    "kuku", "neko", "resta", "kaka",
    "chichi", "kawkaw"
]

plural_animals = {
    "kuku": "kukun",
    "neko": "nekon",
    "resta": "restan",
    "kaka": "kakan",
    "chichi": "chichin",
    "kawkaw": "kawkawn"
}

objects = [
    "vomi", "kafka", "ghar"
]

places = [
    "femi", "damar"
]

# TODO : fix the collision issue with fira(red) and fira(star)(maybe)

nature = [

    "luma", "fira"
]

nouns = people + animals + objects + places + nature

adjectives = [
    "fira", "boku", "sena", "moro",
    "leni", "vaku", "desa", "roni",
    "kema", "zeli", "pira", "nemo",
    "jaku", "henu", "bora", "tenu",
    "fomi", "yera", "dani", "loku"
]

det = ["ta", "na"]


motion = [
    "bhago", "jojo", "davi",
    "nexo", "snan"
]

plural_motion = {
    "bhago": "bhagos",
    "jojo": "jojos",
    "davi": "davis",
    "nexo": "nexos",
    "snan": "snans",
    "icarus": "icaruss"
}

fly = [
    "icarus"
]

perception = [
    "perv",
    "rime"
]

manipulation = [
    "paku",
    "soro",
    "sisy",
    "loti",
    "bobder",
    "fenu",
    "zari"
]

consumption = [
    "yumyum"
]

fly_subjects = [
    "chichi",
    "kawkaw"
]

plural_special = {
    "yumyum": "yumyums",
    "perv": "pervs",
    "rime": "rimes",
    "paku": "pakus",
    "soro": "soros",
    "loti": "lotis",
    "fenu": "fenus",
    "zari": "zaris",
    "sisy": "sisys",
    "bobder": "bobders"
}

eat_subjects = people + animals

read_subjects = people
write_subjects = people

push_subjects = people + animals
push_objects = people + animals + objects



def adj():
    return random.choice(adjectives)

def det_word():
    return random.choice(det)

def noun():
    return random.choice(nouns)


def maybe_plural(noun):
    if noun in plural_animals and random.random() < 0.5:
        return plural_animals[noun], True

    return noun, False


def template_motion():

    subj = random.choice(people + animals)

    subj, is_plural = maybe_plural(subj)

    verb = random.choice(motion)

    if is_plural:
        verb = plural_motion[verb]

    return (
        f"{det_word()} {adj()} {subj} "
        f"{verb}"
    )

def template_fly():

    subj = random.choice(fly_subjects)

    subj, is_plural = maybe_plural(subj)

    verb = "icarus"

    if is_plural:
        verb = "icaruss"

    return (
        f"{det_word()} {adj()} {subj} "
        f"{verb}"
    )


def template_eat():
    subj = random.choice(eat_subjects)
    subj, is_plural = maybe_plural(subj)

    verb = "yumyum"

    if is_plural:
        verb = "yumyums"

    obj = random.choice(
        animals + objects
    )

    return (
        f"{det_word()} {adj()} {subj} "
        f"{verb} "
        f"{det_word()} {adj()} {obj}"
    )

def template_read():
    subj = random.choice(read_subjects)

    return (
        f"{det_word()} {adj()} {subj} "
        f"rime "
        f"{det_word()} {adj()} kafka"
    )

def template_write():
    subj = random.choice(write_subjects)

    return (
        f"{det_word()} {adj()} {subj} "
        f"loti "
        f"{det_word()} {adj()} kafka"
    )

def template_push():
    subj = random.choice(push_subjects)

    obj = random.choice(push_objects)

    return (
        f"{det_word()} {adj()} {subj} "
        f"{random.choice(['paku','soro'])} "
        f"{det_word()} {adj()} {obj}"
    )

def template_see():
    subj = random.choice(people + animals)

    obj = random.choice(nouns)

    return (
        f"{det_word()} {adj()} {subj} "
        f"perv "
        f"{det_word()} {adj()} {obj}"
    )


templates = [
    template_motion,
    template_fly,
    template_eat,
    template_read,
    template_write,
    template_push,
    template_see,
]



sentences = []

for _ in range(100_000):
    sentence = random.choice(templates)()
    sentences.append(sentence)


with open("data/langv2.txt", "w") as f:
    f.write("\n".join(sentences))

print("Generated", len(sentences), "sentences")