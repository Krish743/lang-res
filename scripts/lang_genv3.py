import random
import csv
import json
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

SEED = 1337
NUM_SENTENCES = 200_000

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "langv3.txt"
METADATA_FILE = OUTPUT_DIR / "langv3_metadata.json"
MANIFEST_FILE = OUTPUT_DIR / "langv3_manifest.csv"

random.seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LEXICON
# ============================================================

# ---------------------- Determiners ----------------------

det = ["ta", "na"]  # the / a-an


# ---------------------- People ----------------------

people = [
    "obama",
    "osama",
    "madre",
    "padre",
    "dumdum",
]

plural_people = {
    "obama": "obaman",
    "osama": "osaman",
    "madre": "madren",
    "padre": "padren",
    "dumdum": "dumdumn",
}


# ---------------------- Animals ----------------------

animals = [
    "kuku",
    "neko",
    "resta",
    "kaka",
    "chichi",
    "kawkaw",
    "nori",
]

plural_animals = {
    "kuku": "kukun",
    "neko": "nekon",
    "resta": "restan",
    "kaka": "kakan",
    "chichi": "chichin",
    "kawkaw": "kawkawn",
    "nori": "norin",
}


# ---------------------- Objects ----------------------

objects = [
    "vomi",    # chair
    "kafka",   # book
    "ghar",    # house
    "doro",    # door
    "kito",    # window
    "mepa",    # map
    "pati",    # paper
    "bexa",    # box
]

furniture = [
    "vomi",
]

documents = [
    "kafka",
    "mepa",
    "pati",
]

openable_objects = [
    "doro",
    "kito",
    "bexa",
    "kafka",
]

build_objects = [
    "ghar",
    "doro",
    "kito",
    "bexa",
]

movable_objects = [
    "vomi",
    "kafka",
    "doro",
    "kito",
    "mepa",
    "pati",
    "bexa",
]


# ---------------------- Places / Nature ----------------------

places = [
    "femi",   # mountain
    "damar",  # road
]

nature = [
    "luma",   # star
    "firax",  # fire
]


# ---------------------- Food ----------------------

food = [
    "mira",   # fruit
    "sopa",   # soup
    "bren",   # bread
    "tanu",   # rice
]


# ---------------------- Semantic category labels ----------------------

# These are used only in very rare taxonomy sentences.
category_words = {
    "human": "homa",
    "animal": "zuka",
    "object": "toma",
    "place": "loka",
    "food": "foda",
    "nature": "natura",
}


# ============================================================
# ADJECTIVES
# ============================================================

colors = [
    "fira",   # red
    "boku",   # blue
    "sena",   # green
    "moro",   # yellow
]

living_properties = [
    "leni",   # big
    "vaku",   # small
    "desa",   # fast
    "roni",   # slow
    "kema",   # strong
    "zeli",   # weak
    "fomi",   # happy
    "yera",   # sad
    "dani",   # loud
    "loku",   # quiet
]

physical_properties = [
    "leni",
    "vaku",
    "desa",
    "roni",
    "kema",
    "zeli",
    "pira",   # hot
    "nemo",   # cold
]

object_properties = [
    "leni",
    "vaku",
    "jaku",   # bright
    "henu",   # dark
    "bora",   # new
    "tenu",   # old
    "pira",
    "nemo",
]

place_properties = [
    "leni",
    "vaku",
    "jaku",
    "henu",
    "bora",
    "tenu",
]


# ============================================================
# VERBS
# ============================================================

motion = [
    "bhago",   # run
    "jojo",    # walk
    "davi",    # climb
    "nexo",    # fall
]

swim_verb = "snan"
fly_verb = "icarus"

read_verb = "rime"
write_verb = "loti"

manipulation = [
    "paku",    # push
    "soro",    # pull
    "sisy",    # carry
    "bobder",  # build
    "fenu",    # open
    "zari",    # close
]

eat_verb = "yumyum"
sit_verb = "sita"

# Grammatical markers
relative_marker = "ke"  # who / that
copula = "kesa"          # is / are


# ============================================================
# SEMANTIC COMPATIBILITY
# ============================================================

fly_subjects = {
    "chichi",
    "kawkaw",
}

swim_subjects = {
    "nori",
    "kaka",
    "neko",
    "resta",
}

eat_subjects = set(people + animals)
eat_objects = set(food)

read_subjects = set(people)
read_objects = set(documents)

write_subjects = set(people)
write_objects = set(documents)

build_subjects = set(people)

open_subjects = set(people)
close_subjects = set(people)

carry_subjects = set(people + animals)
carry_objects = set(movable_objects + food)

push_pull_subjects = set(people + animals)
push_pull_objects = set(movable_objects)

see_subjects = set(people + animals)
see_objects = set(
    people
    + animals
    + objects
    + places
    + nature
    + food
)

sit_subjects = set(people + animals)
sit_objects = set(furniture)


# ============================================================
# MORPHOLOGY
# ============================================================

TENSES = [
    "present",
    "past",
    "future",
]


def verb_form(base, plural=False, tense="present"):
    """
    Present:
        singular = base
        plural   = base + s

    Past:
        singular = base + ra
        plural   = base + ras

    Future:
        singular = base + li
        plural   = base + lis
    """

    if tense == "present":
        suffix = "s" if plural else ""

    elif tense == "past":
        suffix = "ras" if plural else "ra"

    elif tense == "future":
        suffix = "lis" if plural else "li"

    else:
        raise ValueError(f"Unknown tense: {tense}")

    return base + suffix


def maybe_plural(noun):
    """
    Only people and animals are pluralized in V3.
    Probability = 45%.
    """

    if noun in plural_people and random.random() < 0.45:
        return plural_people[noun], True

    if noun in plural_animals and random.random() < 0.45:
        return plural_animals[noun], True

    return noun, False


def base_word(form):
    """
    Convert plural human/animal form back to its lexical base.
    """

    reverse = {
        **{v: k for k, v in plural_people.items()},
        **{v: k for k, v in plural_animals.items()},
    }

    return reverse.get(form, form)


def noun_category(noun):
    """
    Return semantic category of noun.
    """

    base = base_word(noun)

    if base in people:
        return "human"

    if base in animals:
        return "animal"

    if base in objects:
        return "object"

    if base in places:
        return "place"

    if base in nature:
        return "nature"

    if base in food:
        return "food"

    return "unknown"


def adjective_for(noun):
    """
    Adjective choice is constrained by semantic type.

    Colors remain broadly compatible.
    Property adjectives depend on noun category.
    """

    category = noun_category(noun)

    if category == "human":
        pool =  living_properties

    elif category == "animal":
        pool = physical_properties + [
            "fomi",
            "yera",
            "dani",
            "loku",
        ]

    elif category == "object":
        pool = colors + object_properties

    elif category == "place":
        pool =  place_properties

    elif category == "nature":
        pool =  [
            "jaku",
            "henu",
            "pira",
            "nemo",
        ]

    elif category == "food":
        pool = colors + [
            "pira",
            "nemo",
            "bora",
            "tenu",
        ]

    else:
        pool = colors

    return random.choice(pool)


def noun_phrase(noun, allow_plural=True):
    """
    Returns:

        phrase
        singular lexical word
        inflected form
        plural flag
        adjective
    """

    if allow_plural:
        noun_form, is_plural = maybe_plural(noun)
    else:
        noun_form, is_plural = noun, False

    adjective = adjective_for(noun_form)

    phrase = (
        f"{random.choice(det)} "
        f"{adjective} "
        f"{noun_form}"
    )

    return (
        phrase,
        base_word(noun_form),
        noun_form,
        is_plural,
        adjective,
    )


def choose_tense():
    return random.choices(
        ["present", "past", "future"],
        weights=[0.60, 0.25, 0.15],
        k=1,
    )[0]


def agreement_form(base_verb, is_plural, tense):
    return verb_form(
        base_verb,
        plural=is_plural,
        tense=tense,
    )


# ============================================================
# SENTENCE TEMPLATES
# ============================================================

def template_motion():

    subj = random.choice(
        people + animals
    )

    (
        subj_phrase,
        subj_base,
        subj_form,
        plural,
        _
    ) = noun_phrase(subj)

    base = random.choice(motion)
    tense = choose_tense()

    verb = agreement_form(
        base,
        plural,
        tense
    )

    return (
        f"{subj_phrase} {verb}",
        {
            "template": "motion",
            "tense": tense,
            "subject": subj_base,
            "subject_category": noun_category(subj_form),
            "subject_plural": plural,
            "object": "",
            "object_category": "",
            "object_plural": False,
        },
    )


def template_fly():

    subj = random.choice(
        list(fly_subjects)
    )

    (
        subj_phrase,
        subj_base,
        subj_form,
        plural,
        _
    ) = noun_phrase(subj)

    tense = choose_tense()

    verb = agreement_form(
        fly_verb,
        plural,
        tense
    )

    return (
        f"{subj_phrase} {verb}",
        {
            "template": "fly",
            "tense": tense,
            "subject": subj_base,
            "subject_category": "animal",
            "subject_plural": plural,
            "object": "",
            "object_category": "",
            "object_plural": False,
        },
    )


def template_swim():

    subj = random.choice(
        list(swim_subjects)
    )

    (
        subj_phrase,
        subj_base,
        subj_form,
        plural,
        _
    ) = noun_phrase(subj)

    tense = choose_tense()

    verb = agreement_form(
        swim_verb,
        plural,
        tense
    )

    return (
        f"{subj_phrase} {verb}",
        {
            "template": "swim",
            "tense": tense,
            "subject": subj_base,
            "subject_category": noun_category(subj_form),
            "subject_plural": plural,
            "object": "",
            "object_category": "",
            "object_plural": False,
        },
    )


def template_eat():

    subj = random.choice(
        list(eat_subjects)
    )

    obj = random.choice(
        list(eat_objects)
    )

    (
        subj_phrase,
        subj_base,
        subj_form,
        subj_plural,
        _
    ) = noun_phrase(subj)

    (
        obj_phrase,
        obj_base,
        obj_form,
        obj_plural,
        _
    ) = noun_phrase(
        obj,
        allow_plural=False
    )

    tense = choose_tense()

    verb = agreement_form(
        eat_verb,
        subj_plural,
        tense
    )

    return (
        f"{subj_phrase} {verb} {obj_phrase}",
        {
            "template": "eat",
            "tense": tense,
            "subject": subj_base,
            "subject_category": noun_category(subj_form),
            "subject_plural": subj_plural,
            "object": obj_base,
            "object_category": "food",
            "object_plural": obj_plural,
        },
    )


def template_read():

    subj = random.choice(
        list(read_subjects)
    )

    obj = random.choice(
        list(read_objects)
    )

    (
        subj_phrase,
        subj_base,
        subj_form,
        plural,
        _
    ) = noun_phrase(subj)

    (
        obj_phrase,
        obj_base,
        obj_form,
        obj_plural,
        _
    ) = noun_phrase(
        obj,
        allow_plural=False
    )

    tense = choose_tense()

    verb = agreement_form(
        read_verb,
        plural,
        tense
    )

    return (
        f"{subj_phrase} {verb} {obj_phrase}",
        {
            "template": "read",
            "tense": tense,
            "subject": subj_base,
            "subject_category": "human",
            "subject_plural": plural,
            "object": obj_base,
            "object_category": noun_category(obj_form),
            "object_plural": obj_plural,
        },
    )


def template_write():

    subj = random.choice(
        list(write_subjects)
    )

    obj = random.choice(
        list(write_objects)
    )

    (
        subj_phrase,
        subj_base,
        subj_form,
        plural,
        _
    ) = noun_phrase(subj)

    (
        obj_phrase,
        obj_base,
        obj_form,
        obj_plural,
        _
    ) = noun_phrase(
        obj,
        allow_plural=False
    )

    tense = choose_tense()

    verb = agreement_form(
        write_verb,
        plural,
        tense
    )

    return (
        f"{subj_phrase} {verb} {obj_phrase}",
        {
            "template": "write",
            "tense": tense,
            "subject": subj_base,
            "subject_category": "human",
            "subject_plural": plural,
            "object": obj_base,
            "object_category": noun_category(obj_form),
            "object_plural": obj_plural,
        },
    )


def template_build():

    subj = random.choice(
        list(build_subjects)
    )

    obj = random.choice(
        build_objects
    )

    (
        subj_phrase,
        subj_base,
        subj_form,
        plural,
        _
    ) = noun_phrase(subj)

    (
        obj_phrase,
        obj_base,
        obj_form,
        obj_plural,
        _
    ) = noun_phrase(
        obj,
        allow_plural=False
    )

    tense = choose_tense()

    verb = agreement_form(
        "bobder",
        plural,
        tense
    )

    return (
        f"{subj_phrase} {verb} {obj_phrase}",
        {
            "template": "build",
            "tense": tense,
            "subject": subj_base,
            "subject_category": "human",
            "subject_plural": plural,
            "object": obj_base,
            "object_category": "object",
            "object_plural": obj_plural,
        },
    )


def template_open_close():

    subj = random.choice(
        list(open_subjects)
    )

    base = random.choice(
        ["fenu", "zari"]
    )

    obj = random.choice(
        openable_objects
    )

    (
        subj_phrase,
        subj_base,
        subj_form,
        plural,
        _
    ) = noun_phrase(subj)

    (
        obj_phrase,
        obj_base,
        obj_form,
        obj_plural,
        _
    ) = noun_phrase(
        obj,
        allow_plural=False
    )

    tense = choose_tense()

    verb = agreement_form(
        base,
        plural,
        tense
    )

    return (
        f"{subj_phrase} {verb} {obj_phrase}",
        {
            "template": "open_close",
            "tense": tense,
            "subject": subj_base,
            "subject_category": "human",
            "subject_plural": plural,
            "object": obj_base,
            "object_category": "object",
            "object_plural": obj_plural,
        },
    )


def template_carry():

    subj = random.choice(
        list(carry_subjects)
    )

    obj = random.choice(
        list(carry_objects)
    )

    (
        subj_phrase,
        subj_base,
        subj_form,
        plural,
        _
    ) = noun_phrase(subj)

    (
        obj_phrase,
        obj_base,
        obj_form,
        obj_plural,
        _
    ) = noun_phrase(
        obj,
        allow_plural=False
    )

    tense = choose_tense()

    verb = agreement_form(
        "sisy",
        plural,
        tense
    )

    return (
        f"{subj_phrase} {verb} {obj_phrase}",
        {
            "template": "carry",
            "tense": tense,
            "subject": subj_base,
            "subject_category": noun_category(subj_form),
            "subject_plural": plural,
            "object": obj_base,
            "object_category": noun_category(obj_form),
            "object_plural": obj_plural,
        },
    )


def template_push_pull():

    subj = random.choice(
        list(push_pull_subjects)
    )

    base = random.choice(
        ["paku", "soro"]
    )

    obj = random.choice(
        list(push_pull_objects)
    )

    (
        subj_phrase,
        subj_base,
        subj_form,
        plural,
        _
    ) = noun_phrase(subj)

    (
        obj_phrase,
        obj_base,
        obj_form,
        obj_plural,
        _
    ) = noun_phrase(
        obj,
        allow_plural=False
    )

    tense = choose_tense()

    verb = agreement_form(
        base,
        plural,
        tense
    )

    return (
        f"{subj_phrase} {verb} {obj_phrase}",
        {
            "template": "push_pull",
            "tense": tense,
            "subject": subj_base,
            "subject_category": noun_category(subj_form),
            "subject_plural": plural,
            "object": obj_base,
            "object_category": noun_category(obj_form),
            "object_plural": obj_plural,
        },
    )


def template_see():

    subj = random.choice(
        list(see_subjects)
    )

    obj = random.choice(
        list(see_objects)
    )

    (
        subj_phrase,
        subj_base,
        subj_form,
        plural,
        _
    ) = noun_phrase(subj)

    (
        obj_phrase,
        obj_base,
        obj_form,
        obj_plural,
        _
    ) = noun_phrase(
        obj,
        allow_plural=False
    )

    tense = choose_tense()

    verb = agreement_form(
        "perv",
        plural,
        tense
    )

    return (
        f"{subj_phrase} {verb} {obj_phrase}",
        {
            "template": "see",
            "tense": tense,
            "subject": subj_base,
            "subject_category": noun_category(subj_form),
            "subject_plural": plural,
            "object": obj_base,
            "object_category": noun_category(obj_form),
            "object_plural": obj_plural,
        },
    )


def template_sit():

    subj = random.choice(
        list(sit_subjects)
    )

    obj = random.choice(
        list(sit_objects)
    )

    (
        subj_phrase,
        subj_base,
        subj_form,
        plural,
        _
    ) = noun_phrase(subj)

    (
        obj_phrase,
        obj_base,
        obj_form,
        obj_plural,
        _
    ) = noun_phrase(
        obj,
        allow_plural=False
    )

    tense = choose_tense()

    verb = agreement_form(
        sit_verb,
        plural,
        tense
    )

    return (
        f"{subj_phrase} {verb} {obj_phrase}",
        {
            "template": "sit",
            "tense": tense,
            "subject": subj_base,
            "subject_category": noun_category(subj_form),
            "subject_plural": plural,
            "object": obj_base,
            "object_category": noun_category(obj_form),
            "object_plural": obj_plural,
        },
    )


def template_relative():
    """
    Relative clause with an omitted object.

    Example:

        ta fira kuku ke na boku neko perv bhago

        = the red dog that a blue cat sees runs

    The main noun ("kuku") is the implicit object
    of the embedded verb "perv".
    """

    main_subj = random.choice(
        people + animals
    )

    embedded_subj = random.choice(
        people + animals
    )

    (
        main_phrase,
        main_base,
        main_form,
        main_plural,
        _
    ) = noun_phrase(main_subj)

    (
        embedded_phrase,
        embedded_base,
        embedded_form,
        embedded_plural,
        _
    ) = noun_phrase(
        embedded_subj
    )

    main_tense = choose_tense()

    embedded_verb = agreement_form(
        "perv",
        embedded_plural,
        "present"
    )

    main_verb = agreement_form(
        random.choice(motion),
        main_plural,
        main_tense
    )

    return (
        f"{main_phrase} "
        f"{relative_marker} "
        f"{embedded_phrase} "
        f"{embedded_verb} "
        f"{main_verb}",

        {
            "template": "relative",
            "tense": main_tense,
            "subject": main_base,
            "subject_category": noun_category(main_form),
            "subject_plural": main_plural,
            "object": embedded_base,
            "object_category": noun_category(embedded_form),
            "object_plural": embedded_plural,
        },
    )


def template_nested_relative():
    """
    Rare nested relative clause.

    Rough structure:

        MAIN
        ke EMBEDDED
        ke DEEP
        DEEP_VERB
        EMBEDDED_VERB
        MAIN_VERB
    """

    main_subj = random.choice(
        people + animals
    )

    embedded_subj = random.choice(
        people + animals
    )

    deep_subj = random.choice(
        people + animals
    )

    (
        main_phrase,
        main_base,
        main_form,
        main_plural,
        _
    ) = noun_phrase(main_subj)

    (
        embedded_phrase,
        embedded_base,
        embedded_form,
        embedded_plural,
        _
    ) = noun_phrase(embedded_subj)

    (
        deep_phrase,
        deep_base,
        deep_form,
        deep_plural,
        _
    ) = noun_phrase(deep_subj)

    deep_verb = agreement_form(
        "perv",
        deep_plural,
        "present"
    )

    embedded_verb = agreement_form(
        "perv",
        embedded_plural,
        "present"
    )

    main_verb = agreement_form(
        random.choice(motion),
        main_plural,
        choose_tense()
    )

    return (
        f"{main_phrase} "
        f"{relative_marker} "
        f"{embedded_phrase} "
        f"{relative_marker} "
        f"{deep_phrase} "
        f"{deep_verb} "
        f"{embedded_verb} "
        f"{main_verb}",

        {
            "template": "nested_relative",
            "tense": "mixed",
            "subject": main_base,
            "subject_category": noun_category(main_form),
            "subject_plural": main_plural,
            "object": embedded_base,
            "object_category": noun_category(embedded_form),
            "object_plural": embedded_plural,
        },
    )


def template_taxonomy():
    """
    Rare explicit semantic hierarchy examples.

        ta kuku kesa ta zuka
        the dog is an animal

    These are deliberately rare so the model isn't simply
    handed all semantic categories explicitly.
    """

    base_noun = random.choice(
        people
        + animals
        + objects
        + places
        + food
        + nature
    )

    category = noun_category(base_noun)

    category_noun = category_words[category]

    (
        subj_phrase,
        subj_base,
        subj_form,
        plural,
        _
    ) = noun_phrase(
        base_noun,
        allow_plural=False
    )

    return (
        f"{subj_phrase} "
        f"{copula} "
        f"ta {category_noun}",

        {
            "template": "taxonomy",
            "tense": "present",
            "subject": subj_base,
            "subject_category": category,
            "subject_plural": plural,
            "object": category_noun,
            "object_category": "category_label",
            "object_plural": False,
        },
    )


# ============================================================
# WEIGHTED TEMPLATE DISTRIBUTION
# ============================================================

template_specs = [
    (template_motion, 0.14),
    (template_fly, 0.05),
    (template_swim, 0.05),
    (template_eat, 0.14),
    (template_read, 0.09),
    (template_write, 0.07),
    (template_build, 0.07),
    (template_open_close, 0.06),
    (template_carry, 0.06),
    (template_push_pull, 0.10),
    (template_see, 0.10),
    (template_sit, 0.05),
    (template_relative, 0.02),
]

TEMPLATE_FUNCS = [
    fn
    for fn, _ in template_specs
]

TEMPLATE_WEIGHTS = [
    weight
    for _, weight in template_specs
]

# Rare high-complexity examples
NESTED_PROB = 0.05
TAXONOMY_PROB = 0.01


# ============================================================
# GENERATE DATASET
# ============================================================

sentences = []
manifest = []
seen = set()

MAX_ATTEMPTS = NUM_SENTENCES * 20
attempts = 0

while (
    len(sentences) < NUM_SENTENCES
    and attempts < MAX_ATTEMPTS
):

    attempts += 1

    r = random.random()

    if r < NESTED_PROB:

        sentence, info = (
            template_nested_relative()
        )

    elif r < NESTED_PROB + TAXONOMY_PROB:

        sentence, info = (
            template_taxonomy()
        )

    else:

        fn = random.choices(
            TEMPLATE_FUNCS,
            weights=TEMPLATE_WEIGHTS,
            k=1,
        )[0]

        sentence, info = fn()

    # Keep exact duplicates out of the final corpus.
    if sentence in seen:
        continue

    seen.add(sentence)

    record = {
        "sentence_id": len(sentences),
        "sentence": sentence,
        "token_count": len(sentence.split()),
        **info,
    }

    sentences.append(sentence)
    manifest.append(record)


if len(sentences) < NUM_SENTENCES:
    raise RuntimeError(
        f"Could only generate "
        f"{len(sentences):,} unique sentences "
        f"after {attempts:,} attempts."
    )


# ============================================================
# METADATA
# ============================================================

lexicon = {

    "determiners": {
        "ta": "the",
        "na": "a/an",
    },

    "people": {
        "obama": "man",
        "osama": "woman",
        "madre": "mother",
        "padre": "father",
        "dumdum": "student",
    },

    "animals": {
        "kuku": "dog",
        "neko": "cat",
        "resta": "wolf",
        "kaka": "crocodile",
        "chichi": "sparrow",
        "kawkaw": "crow",
        "nori": "fish",
    },

    "objects": {
        "vomi": "chair",
        "kafka": "book",
        "ghar": "house",
        "doro": "door",
        "kito": "window",
        "mepa": "map",
        "pati": "paper",
        "bexa": "box",
    },

    "places": {
        "femi": "mountain",
        "damar": "road",
    },

    "nature": {
        "luma": "star",
        "firax": "fire",
    },

    "food": {
        "mira": "fruit",
        "sopa": "soup",
        "bren": "bread",
        "tanu": "rice",
    },

    "adjectives": {
        "fira": "red",
        "boku": "blue",
        "sena": "green",
        "moro": "yellow",
        "leni": "big",
        "vaku": "small",
        "desa": "fast",
        "roni": "slow",
        "kema": "strong",
        "zeli": "weak",
        "pira": "hot",
        "nemo": "cold",
        "jaku": "bright",
        "henu": "dark",
        "bora": "new",
        "tenu": "old",
        "fomi": "happy",
        "yera": "sad",
        "dani": "loud",
        "loku": "quiet",
    },

    "verbs": {
        "bhago": "run",
        "jojo": "walk",
        "davi": "climb",
        "nexo": "fall",
        "snan": "swim",
        "icarus": "fly",
        "perv": "see",
        "rime": "read",
        "loti": "write",
        "paku": "push",
        "soro": "pull",
        "sisy": "carry",
        "bobder": "build",
        "fenu": "open",
        "zari": "close",
        "yumyum": "eat",
        "sita": "sit",
    },

    "special": {
        "ke": "who/that",
        "kesa": "is/are",
    },
}

metadata = {
    "language": "LangV3",
    "seed": SEED,
    "num_sentences": NUM_SENTENCES,

    "morphology": {
        "plural_noun_suffix": "n",
        "plural_verb_suffix": "s",
        "past_suffix": "ra",
        "future_suffix": "li",
    },

    "lexicon": lexicon,

    "category_words": category_words,

    "generation": {
        "nested_relative_probability": NESTED_PROB,
        "taxonomy_probability": TAXONOMY_PROB,

        "template_weights": {
            fn.__name__: weight
            for fn, weight in template_specs
        },
    },
}


# ============================================================
# SAVE
# ============================================================

with OUTPUT_FILE.open(
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(sentences)
    )


with METADATA_FILE.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


with MANIFEST_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    fieldnames = [
        "sentence_id",
        "sentence",
        "token_count",
        "template",
        "tense",
        "subject",
        "subject_category",
        "subject_plural",
        "object",
        "object_category",
        "object_plural",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(manifest)


# ============================================================
# SUMMARY
# ============================================================

print(
    f"Generated "
    f"{len(sentences):,} unique sentences."
)

print(f"Corpus   : {OUTPUT_FILE}")
print(f"Metadata : {METADATA_FILE}")
print(f"Manifest : {MANIFEST_FILE}")

print("\nExamples:")

for name, fn in [
    ("motion", template_motion),
    ("fly", template_fly),
    ("swim", template_swim),
    ("eat", template_eat),
    ("read", template_read),
    ("build", template_build),
    ("open/close", template_open_close),
    ("relative", template_relative),
    ("nested", template_nested_relative),
    ("taxonomy", template_taxonomy),
]:
    print(
        f"{name:12s} -> {fn()[0]}"
    )