"""
Runyoro-Rutooro language rules.
Sources:
  - A Grammar of Runyoro-Rutooro (Chapters 2, 4, 7, 13, 15, 16)
  - Runyoro-Rutooro Orthography Guide (Ministry of Gender, Uganda, 1995)
  - runyorodictionary.com
"""
import re as _re

# ─────────────────────────────────────────────────────────────────────────────
# ALPHABET & ORTHOGRAPHY
# Source: Runyoro-Rutooro Orthography Guide (1995)
# ─────────────────────────────────────────────────────────────────────────────

ALPHABET = list("abcdefghijklmnoprstuwyz")
# Letters NOT in Runyoro-Rutooro: q, v, x
ABSENT_LETTERS = {"q", "v", "x"}

VOWELS = {"a", "e", "i", "o", "u"}

# Vowel length: doubled vowel = long vowel
# e.g. aa, ee, ii, oo, uu indicate long vowels
LONG_VOWEL_PATTERN = _re.compile(r'([aeiou])\1', _re.IGNORECASE)

# Diphthongs common in Runyoro-Rutooro
DIPHTHONGS = {"ai", "oi", "ei", "au", "ou"}

# Apostrophe rule: used when initial vowel of a word is swallowed in fast speech
# e.g.  n' + ente  -> n'ente,  z' + ente -> z'ente
APOSTROPHE_CONTEXTS = [
    ("n'", "with/and (before nouns starting with vowel)"),
    ("z'", "of (class 10, before vowel-initial nouns)"),
    ("k'", "it is (before vowel)"),
    ("y'", "of (class 9, before vowel)"),
    ("w'", "of (class 1, before vowel)"),
    ("g'", "of (class 3/6, before vowel)"),
    ("b'", "of (class 2, before vowel)"),
]

# ─────────────────────────────────────────────────────────────────────────────
# R / L RULE
# Source: Orthography Guide + Grammar Ch.2
# ─────────────────────────────────────────────────────────────────────────────

RL_RULE = (
    "R/L Rule: In Runyoro-Rutooro, 'R' is the dominant consonant. "
    "'L' is only used immediately before or after the vowels 'e' or 'i'. "
    "In all other positions 'R' is used instead of 'L'."
)

def apply_rl_rule(text: str) -> str:
    """Replace L with R except when adjacent to e or i."""
    if not text:
        return text
    chars = list(text)
    result = []
    for i, ch in enumerate(chars):
        if ch not in ('l', 'L'):
            result.append(ch)
            continue
        prev = chars[i - 1].lower() if i > 0 else ''
        nxt  = chars[i + 1].lower() if i < len(chars) - 1 else ''
        if prev in ('e', 'i') or nxt in ('e', 'i'):
            result.append(ch)
        else:
            result.append('R' if ch.isupper() else 'r')
    return ''.join(result)


# ─────────────────────────────────────────────────────────────────────────────
# SOUND CHANGE RULES
# Source: Grammar Ch.2 — Sound change in vowels and consonants
# ─────────────────────────────────────────────────────────────────────────────

# Consonant + suffix transformations (applied to verb stems)
# r/t/j + -ire/-ere/-i/-ya undergo the following changes:
CONSONANT_SUFFIX_CHANGES = {
    # (stem_final_consonant, suffix) -> result
    ("r",  "-ire"):  "-zire",
    ("r",  "-i"):    "-zi",
    ("r",  "-ya"):   "-za",
    ("t",  "-ire"):  "-sire",
    ("t",  "-i"):    "-si",
    ("t",  "-ya"):   "-sa",
    ("j",  "-ire"):  "-zire",
    ("j",  "-i"):    "-zi",
    ("nd", "-ire"):  "-nzire",
    ("nd", "-i"):    "-nzi",
    ("nd", "-ya"):   "-nza",
    ("nt", "-ire"):  "-nsire",
    ("nt", "-i"):    "-nsi",
    ("nt", "-ya"):   "-nsa",
}

# Nasal assimilation: n before bilabials b/p -> m
# n + b -> mb,  n + m -> mm
NASAL_ASSIMILATION = {
    "nb": "mb",
    "np": "mp",
    "nm": "mm",
    "nr": "nd",   # n + r -> nd (Meinhof's rule)
    "nl": "nd",
}

# Present imperfect prefix ni- vowel change before certain concords
# ni + u-class concord -> nu
# e.g. nimugenda -> numugenda, niguteera -> nuguteera
NI_PREFIX_CHANGE = {
    "nimu": "numu",
    "nigu": "nugu",
    "niru": "nuru",
    "nibu": "nubu",
    "nikw": "nukw",
}

# Y-insertion rule: after tense prefixes a-, ra-, raa- with subject prefixes,
# y is inserted before verb stems beginning with a vowel
# e.g. a + ira -> ayira,  ra + ira -> rayira
Y_INSERTION_PREFIXES = {"a", "ra", "raa", "daa"}

# Reflexive verb imperatives: stem begins with e but singular imperative uses w + long vowel
# e.g. okw-esereka -> weesereke (hide yourself)
REFLEXIVE_IMPERATIVE_PREFIX = "wee"
REFLEXIVE_PLURAL_PREFIX = "mwe"

# Conversive verb suffix rule:
# If simple stem vowel is a/e/i/o -> conversive suffix starts with u
# If simple stem vowel is o (long oo) -> conversive suffix starts with o
CONVERSIVE_SUFFIX = {
    "a": "ura", "e": "ura", "i": "ura", "u": "ura",
    "o": "ora",  # long o
}

# ─────────────────────────────────────────────────────────────────────────────
# NOUN CLASS SYSTEM  (Classes 1–15)
# Source: Grammar Ch.7 — The noun class system
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: class_number -> {prefix, plural_class, plural_prefix, description}
NOUN_CLASSES = {
    1:  {"sg_prefix": "omu-",  "sg_prefix_v": "omw-",  "pl_class": 2,  "pl_prefix": "aba-",  "pl_prefix_v": "ab-",   "desc": "persons (singular)"},
    2:  {"sg_prefix": "aba-",  "sg_prefix_v": "ab-",   "pl_class": 1,  "pl_prefix": "omu-",  "pl_prefix_v": "omw-",  "desc": "persons (plural)"},
    "1a": {"sg_prefix": "",    "pl_prefix": "baa-",    "desc": "proper names, titles (no prefix)"},
    "2a": {"sg_prefix": "baa-","pl_prefix": "",         "desc": "plural of class 1a proper names"},
    3:  {"sg_prefix": "emi-",  "sg_prefix_v": "emy-",  "pl_class": 4,  "pl_prefix": "emi-",  "desc": "trees, plants, body parts (singular)"},
    4:  {"sg_prefix": "emi-",  "pl_class": 3,  "pl_prefix": "omu-", "desc": "plural of class 3"},
    5:  {"sg_prefix": "eri-",  "sg_prefix_v": "ery-",  "pl_class": 6,  "pl_prefix": "ama-",  "pl_prefix_e": "ame-", "pl_prefix_o": "amo-", "desc": "augmentatives, some body parts (singular)"},
    6:  {"sg_prefix": "ama-",  "sg_prefix_e": "ame-",  "sg_prefix_o": "amo-", "pl_class": 5, "desc": "plural of class 5; also plural of classes 9/11/14/15"},
    7:  {"sg_prefix": "eki-",  "sg_prefix_v": "eky-",  "pl_class": 8,  "pl_prefix": "ebi-",  "pl_prefix_v": "eby-",  "desc": "things, abstracts, diminutives (singular)"},
    8:  {"sg_prefix": "ebi-",  "sg_prefix_v": "eby-",  "pl_class": 7,  "pl_prefix": "eki-",  "pl_prefix_v": "eky-",  "desc": "plural of class 7"},
    9:  {"sg_prefix": "en-",   "sg_prefix_b": "em-",   "pl_class": 10, "pl_prefix": "en-",   "desc": "animals, foreign words (singular); prefix en- before consonants, em- before b/p"},
    10: {"sg_prefix": "en-",   "sg_prefix_b": "em-",   "pl_class": 9,  "desc": "plural of class 9; also plural of class 11"},
    "9a": {"sg_prefix": "",    "pl_class": "10a", "pl_prefix": "zaa-", "desc": "foreign words, colours, animal names, place names (no prefix, no initial vowel)"},
    "10a":{"sg_prefix": "zaa-","sg_prefix_a": "za-", "sg_prefix_e": "ze-", "sg_prefix_o": "zo-", "desc": "plural of class 9a"},
    11: {"sg_prefix": "oru-",  "sg_prefix_v": "orw-",  "pl_class": 10, "pl_prefix": "en-",   "desc": "long/thin objects, languages, abstract (singular)"},
    12: {"sg_prefix": "aka-",  "sg_prefix_v": "akw-",  "pl_class": 13, "pl_prefix": "utu-",  "desc": "diminutives (singular)"},
    13: {"sg_prefix": "utu-",  "sg_prefix_v": "utw-",  "pl_class": 12, "pl_prefix": "aka-",  "desc": "diminutives (plural) / small quantities"},
    14: {"sg_prefix": "obu-",  "sg_prefix_v": "obw-",  "pl_class": 6,  "pl_prefix": "ama-",  "desc": "abstract nouns, mass nouns"},
    15: {"sg_prefix": "oku-",  "sg_prefix_v": "okw-",  "pl_class": 6,  "pl_prefix": "ama-",  "desc": "verbal infinitives, body parts"},
}

# Prefix -> class lookup (for morphological analysis)
PREFIX_TO_CLASS: dict[str, list] = {
    "omu": [1], "omw": [1], "aba": [2], "ab": [2],
    "emi": [3, 4], "emy": [3],
    "eri": [5], "ery": [5], "ama": [6], "ame": [6], "amo": [6],
    "eki": [7], "eky": [7], "ebi": [8], "eby": [8],
    "en":  [9, 10], "em": [9, 10],
    "oru": [11], "orw": [11],
    "aka": [12], "akw": [12],
    "utu": [13], "utw": [13],
    "obu": [14], "obw": [14],
    "oku": [15], "okw": [15],
    "zaa": ["10a"], "za": ["10a"], "ze": ["10a"], "zo": ["10a"],
    "baa": ["2a"],
}

def get_noun_class(word: str) -> list[int | str]:
    """Return probable noun class(es) for a Runyoro-Rutooro word based on prefix."""
    w = word.lower().strip()
    # Try matching with initial vowel first, then without
    for candidate in [w, w[1:] if w and w[0] in ('a', 'e', 'o') else w]:
        for prefix in sorted(PREFIX_TO_CLASS.keys(), key=len, reverse=True):
            if candidate.startswith(prefix):
                return PREFIX_TO_CLASS[prefix]
    return []

# ─────────────────────────────────────────────────────────────────────────────
# CONCORDIAL AGREEMENT
# Source: Grammar Ch.7 — noun class concordial prefixes
# Each noun class has its own subject concord, object concord, adjective concord
# ─────────────────────────────────────────────────────────────────────────────

# class -> (subject_concord, object_concord, adjective_concord, demonstrative)
CONCORDIAL_AGREEMENT = {
    1:    ("a-",   "-mu-",  "omu-",  "uyu"),
    2:    ("ba-",  "-ba-",  "aba-",  "aba"),
    "1a": ("a-",   "-mu-",  "",      "uyu"),
    "2a": ("ba-",  "-ba-",  "baa-",  "aba"),
    3:    ("gu-",  "-gu-",  "omu-",  "ogwo"),
    4:    ("gi-",  "-gi-",  "emi-",  "egi"),
    5:    ("li-",  "-li-",  "eri-",  "eryo"),
    6:    ("ga-",  "-ga-",  "ama-",  "ago"),
    7:    ("ki-",  "-ki-",  "eki-",  "ekyo"),
    8:    ("bi-",  "-bi-",  "ebi-",  "ebyo"),
    9:    ("i-",   "-i-",   "en-",   "eno"),
    10:   ("zi-",  "-zi-",  "en-",   "ezo"),
    "9a": ("i-",   "-i-",   "",      "eno"),
    "10a":("zi-",  "-zi-",  "",      "ezo"),
    11:   ("ru-",  "-ru-",  "oru-",  "orwo"),
    12:   ("ka-",  "-ka-",  "aka-",  "ako"),
    13:   ("tu-",  "-tu-",  "utu-",  "utu"),
    14:   ("bu-",  "-bu-",  "obu-",  "obwo"),
    15:   ("ku-",  "-ku-",  "oku-",  "okwo"),
}

def get_subject_concord(noun_class: int | str) -> str:
    entry = CONCORDIAL_AGREEMENT.get(noun_class)
    return entry[0] if entry else ""

def get_object_concord(noun_class: int | str) -> str:
    entry = CONCORDIAL_AGREEMENT.get(noun_class)
    return entry[1] if entry else ""


# ─────────────────────────────────────────────────────────────────────────────
# PLURAL FORMATION
# Source: Grammar Ch.7
# ─────────────────────────────────────────────────────────────────────────────

# Common sound changes in plural formation (class 11 -> class 10)
# oru- prefix drops, nasal prefix en-/em- added, with internal sound changes
PLURAL_SOUND_CHANGES = {
    "orubengo":  "emengo",   # lower millstone
    "orulimi":   "endimi",   # tongue/language
    "orugoye":   "engoye",   # cloth
    "orubabi":   "embabi",   # plantain leaf
    "orubaju":   "embaju",   # side/rib
    "orupapura": "empapura", # paper
    "oruseke":   "enseke",   # tube/pipe
    "orubango":  "emango",   # shaft of spear
    "oruhara":   "empara",   # baldness
    "orumuli":   "emuli",    # reed torch
    "orunwa":    "enwa",     # beak/bill
    "orunaku":   "enaku",    # stinging centipede
    "orunumbu":  "enumbu",   # edible tuber
    "orunyaanya":"enyaanya", # tomato
}

# Class 5 -> Class 6 plural prefix rules
# ama- before consonant and vowels a/i
# ame- before e,  amo- before o
def get_class6_prefix(stem: str) -> str:
    if not stem:
        return "ama"
    first = stem[0].lower()
    if first == 'e':
        return "ame"
    if first == 'o':
        return "amo"
    return "ama"

# ─────────────────────────────────────────────────────────────────────────────
# VERB STRUCTURE
# Source: Grammar Ch.4, Ch.13
# ─────────────────────────────────────────────────────────────────────────────

# Infinitive prefix: oku- (before consonant), okw- (before vowel)
INFINITIVE_PREFIX = "oku"
INFINITIVE_PREFIX_V = "okw"

# Subject prefixes (personal pronouns as verb prefixes)
SUBJECT_PREFIXES = {
    "1sg":  "n-",    # I
    "2sg":  "o-",    # you (sg)
    "3sg":  "a-",    # he/she
    "1pl":  "tu-",   # we
    "2pl":  "mu-",   # you (pl)
    "3pl":  "ba-",   # they
}

# Tense/aspect markers (inserted between subject prefix and verb stem)
TENSE_MARKERS = {
    "present_imperfect":  "ni-",   # ongoing action: nigenda = is going
    "recent_past":        "a-",    # just now: nayara = I just made the bed
    "remote_past":        "ka-",   # past: nkaara = I made the bed
    "future":             "ra-",   # future: ndaayara = I shall make the bed
    "perfect":            "i-",    # present perfect: nkozire = I have done
    "negative":           "ti-",   # negation prefix
    "habitual":           "mara-", # habitual/always
}

# Negative tense forms
NEGATIVE_MARKERS = {
    "present":      "ti-",
    "past":         "tinka-",
    "future":       "tinda-",
    "perfect":      "tinka-...-ire",
}

# Common verb suffixes
VERB_SUFFIXES = {
    "-ire / -ere":  "perfect tense",
    "-a":           "simple present / infinitive base",
    "-aho":         "completive (action done at a place)",
    "-anga":        "habitual/frequentative",
    "-isa / -esa":  "causative",
    "-ibwa / -ebwa":"passive",
    "-ana":         "reciprocal (each other)",
    "-ura / -ora":  "conversive/reversive (undo the action)",
    "-uka / -oka":  "intransitive conversive",
    "-rra":         "intensive/completive",
}

# Derivative verb suffixes
# Source: Grammar Ch.12 — Derivative verbs
DERIVATIVE_SUFFIXES = {
    "causative":      ["-isa", "-esa", "-ya"],
    "passive":        ["-ibwa", "-ebwa", "-wa"],
    "reciprocal":     ["-ana"],
    "reversive":      ["-ura", "-ora", "-ula", "-ola"],
    "neuter":         ["-uka", "-oka"],
    "intensive":      ["-rra", "-rruka", "-rrura"],
    "applied":        ["-era", "-ira"],   # action done for/at
    "positional":     ["-ama"],           # be in a position
}


# ─────────────────────────────────────────────────────────────────────────────
# TENSE SYSTEM SUMMARY
# Source: Grammar Ch.13, Ch.15
# ─────────────────────────────────────────────────────────────────────────────

TENSES = {
    "present_imperfect":    {"marker": "ni-",    "example": "nigenda",      "meaning": "is going"},
    "present_perfect":      {"marker": "-ire",   "example": "agenzire",     "meaning": "has gone"},
    "recent_past":          {"marker": "a-",     "example": "nayara",       "meaning": "just now I made the bed"},
    "remote_past":          {"marker": "ka-",    "example": "nkaara",       "meaning": "I made the bed (remote)"},
    "future_immediate":     {"marker": "ra-",    "example": "ndaayara",     "meaning": "I shall make the bed"},
    "future_remote":        {"marker": "raa-",   "example": "turaayara",    "meaning": "we shall make the bed"},
    "conditional":          {"marker": "-ku-",   "example": "obaire okukora","meaning": "if/when (conditional)"},
    "imperative_sg":        {"marker": "stem-a", "example": "genda",        "meaning": "go! (singular)"},
    "imperative_pl":        {"marker": "mu-stem-e","example": "mugende",    "meaning": "go! (plural)"},
    "negative_present":     {"marker": "ti-ni-", "example": "tinigenda",    "meaning": "is not going"},
    "negative_perfect":     {"marker": "tinka-", "example": "tinkagenzire", "meaning": "has not gone"},
}

# Conditional tense: obu/kuba + -ku- prefix
CONDITIONAL_PARTICLES = ["obu", "kuba", "kakuba", "kusangwa", "kakusangwa"]

# ─────────────────────────────────────────────────────────────────────────────
# ADJECTIVES & ADVERBS
# Source: Grammar Ch.16
# ─────────────────────────────────────────────────────────────────────────────

# Adjectives agree with noun class via adjectival concord
# Comparison uses the verb "okusinga" (to surpass/exceed)
COMPARISON = {
    "positive":     "adjective alone — e.g. omukazi omurungi (a good woman)",
    "comparative":  "verb + okusinga — e.g. asinga omurungi (she is better)",
    "superlative":  "verb + okusinga + bose/byona — e.g. asinga bose omurungi (she is the best)",
}

# Common adjective stems (take adjectival concord prefix per noun class)
ADJECTIVE_STEMS = {
    "-rungi":    "good",
    "-bi":       "bad",
    "-raira":    "tall/long",
    "-to":       "small/young",
    "-nene":     "big/fat",
    "-gu":       "heavy",
    "-eri":      "two",
    "-satu":     "three",
    "-na":       "four",
    "-taano":    "five",
    "-ingi":     "many",
    "-eke":      "few/little",
    "-iza":      "good/beautiful (alternative)",
    "-ire":      "old (of things)",
    "-kuru":     "old (of persons)",
}

# Adverbs of manner: formed with ki- prefix or standalone
ADVERBS_OF_MANNER = {
    "kijungu":      "in a European fashion",
    "kiserukali":   "like a soldier",
    "kinyoro":      "like a chief / in Runyoro fashion",
    "kizaana":      "like a maid-servant",
    "masaija":      "in a manly way",
    "mate":         "in a cow-like fashion",
    "matale":       "in a leonine fashion",
    "bwangu":       "quickly, rapidly",
    "mpola":        "slowly, gently",
    "nkoomu":       "together",
    "hamwe":        "together",
}


# ─────────────────────────────────────────────────────────────────────────────
# NUMBERS (Okubara)
# Source: Grammar Ch.4 + Grammar numbers chapter
# ─────────────────────────────────────────────────────────────────────────────

NUMBERS = {
    1: "emu",    2: "ibiri",   3: "isatu",  4: "ina",    5: "itaano",
    6: "mukaaga",7: "musanju", 8: "munaana",9: "mwenda", 10: "ikumi",
    11: "ikumi nemu",          20: "abiri",  30: "asatu",
    40: "ana",   50: "atano",  60: "nkaaga", 70: "nsanju",
    80: "kinaana", 90: "kyenda",
    100: "kikumi", 200: "bibiri", 300: "bisatu", 400: "bina",
    1000: "rukumi", 1_000_000: "akakaikuru", 1_000_000_000: "akasirira",
}

# Hundreds/thousands connected by "mu" and "na"
# e.g. 235 cows = ente bibiri mu asatu na itaano
# Ordinals formed by adding numeral concord to stem
ORDINAL_NOTE = (
    "Ordinals are formed by bringing the numeral into concordial agreement "
    "with the noun it qualifies, using the numeral concord for that class."
)

# Numbers 1-5 must agree with noun class via numeral concord
NUMERAL_CONCORDS = {
    # class: concord_prefix
    1:    "omu",  2:  "aba",
    3:    "omu",  4:  "emi",
    5:    "eri",  6:  "ama",
    7:    "eki",  8:  "ebi",
    9:    "en",   10: "en",
    11:   "oru",  12: "aka",
    13:   "utu",  14: "obu",
    15:   "oku",
}

# ─────────────────────────────────────────────────────────────────────────────
# PARTICLES, CONJUNCTIONS & PREPOSITIONS
# Source: Grammar Ch.4
# ─────────────────────────────────────────────────────────────────────────────

CONJUNCTIONS = {
    "na":           "and / with",
    "hamwe na":     "together with",
    "rundi":        "or / either...or",
    "kandi":        "and / but / in addition",
    "ekindi":       "in addition to",
    "kuba":         "because / that / if",
    "kakuba":       "if (negative conditional)",
    "ngu":          "that (reported speech)",
    "obu":          "if / when",
    "noobwa":       "even if / though / although",
    "kyonka":       "but",
    "baitwa":       "but / whereas",
    "nikyo kinu":   "all the same",
}

PREPOSITIONS = {
    "mu":       "in / into / at",
    "ha":       "at / on / near",
    "ku":       "to / at / on",
    "aha":      "at / there",
    "hanyuma":  "after",
    "nka":      "like / as",
    "okuhikya": "till / until",
    "nkoomu":   "as / like",
    "okuna":    "with / by",
}

NEGATION_WORDS = {
    "ti-":      "negative prefix (verb)",
    "tindi":    "I will not",
    "tinka":    "I did not / have not",
    "aha":      "not there (declinable negation)",
    "busa":     "no / not at all",
    "nga":      "no / not",
}

# Relative particle -nya- / nya-
# Placed before noun prefix to indicate something already known to both speaker and listener
NYA_PARTICLE = {
    "rule": "nya- placed before noun prefix indicates definiteness / already known referent",
    "examples": {
        "nyamotoka": "those (specific) cars",
        "nyastookingi": "those (specific) stockings",
    },
    "similar_to": "zaa-/za-/ze-/zo- prefixes of class 10a",
}


# ─────────────────────────────────────────────────────────────────────────────
# PRONOUNS
# Source: Grammar Ch.4
# ─────────────────────────────────────────────────────────────────────────────

PERSONAL_PRONOUNS = {
    "nyowe":  "I (emphatic)",
    "itwe":   "we (emphatic)",
    "iwe":    "you (singular)",
    "inywe":  "you (plural)",
    "uwe":    "he / she",
}

# Object pronouns (as suffixes/infixes in verb)
OBJECT_PRONOUNS = {
    "1sg": "-ndi- / -m-",
    "2sg": "-ku-",
    "3sg": "-mu-",
    "1pl": "-tu-",
    "2pl": "-ba- (mu-)",
    "3pl": "-ba-",
}


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE NAMES (oru- prefix)
# Source: Grammar Ch.7 — Class 11 semantic properties
# ─────────────────────────────────────────────────────────────────────────────

LANGUAGE_NAMES = {
    "Orunyoro":     "language of Bunyoro",
    "Orutooro":     "language of Tooro",
    "Oruganda":     "language of Buganda (Luganda)",
    "Orunyankole":  "language of Nkole (Ankole)",
    "Orungereza":   "language of England (English)",
    "Orufuransa":   "language of France (French)",
    "Oruswahili":   "Swahili",
    "Oruarabu":     "Arabic",
}


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATIVE / PEJORATIVE PREFIX SUBSTITUTION
# Source: Grammar Ch.7
# ─────────────────────────────────────────────────────────────────────────────

# oru- substituted for normal class prefix = augmentative or pejorative
AUGMENTATIVE_EXAMPLES = {
    "orusaija":  ("omusaija",  "man",    "clumsy/big man (pejorative)"),
    "orukazi":   ("omukazi",   "woman",  "clumsy woman"),
    "orwisiki":  ("omwisiki",  "girl",   "clumsy girl"),
    "orute":     ("ente",      "cow",    "clumsy cow"),
    "oruti":     ("omuti",     "tree",   "long stick"),
    "orunyonyi": ("enyonyi",   "bird",   "big long bird"),
}

# eki-/eky- substituted = magnitude, affection, or contempt
MAGNITUDE_EXAMPLES = {
    "ekisaija":  ("omusaija",  "man",    "that clumsy/big man (contempt)"),
    "ekiiru":    ("omwiru",    "servant","dear poor man (affection) / sturdy peasant"),
    "ekintu":    ("okintu",    "thing",  "monster-like thing"),
}

# eri-/ery- substituted = magnitude
MAGNITUDE_ERI_EXAMPLES = {
    "eriiru":    ("omwiru",    "servant","that sturdy peasant"),
    "erintu":    ("okintu",    "thing",  "monster-like thing"),
    "eryana":    ("omwana",    "child",  "insolent child"),
}

# ─────────────────────────────────────────────────────────────────────────────
# EMPAAKO (Honorific Names)
# ─────────────────────────────────────────────────────────────────────────────

EMPAAKO = {
    "Atwooki":  "From Atwok — shining star. Given to a child born on a night with stars.",
    "Ateenyi":  "From Ateng — beautiful. Given if parents believe the child is beautiful.",
    "Apuuli":   "From Apul/Rapuli — a very lovely girl, center of attraction and love in the family.",
    "Amooti":   "From Amot — princely, a sign of royalty. Mostly for sons and daughters of kings and chiefs.",
    "Akiiki":   "From Ochii/Achii — the one who follows twins. Mostly for firstborns who bear responsibility for siblings.",
    "Adyeeri":  "From adyee/odee — parents had failed to get a child and only got one after spiritual intervention.",
    "Acaali":   "From Ochal — replica. Given to a child who resembled someone in the family or ancestors.",
    "Abaala":   "From Obal/Abal — destroyer/warrior. Usually for sons of chiefs.",
    "Abbooki":  "From Obok/Abok — beloved. Child born out of strong love between parents.",
    "Araali":   "From Arali/Olal/Alal — lost. Given to the only surviving child after mother lost many children.",
    "Abwooli":  "From Abwolo — 'I lied to you'. A woman who conceives but continues having periods.",
    "Okaali":   "From 'kal' — royalty. Only for the King in Kitara customs.",
}


# ─────────────────────────────────────────────────────────────────────────────
# INTERJECTIONS (Ebihunaazo)
# ─────────────────────────────────────────────────────────────────────────────

INTERJECTIONS = {
    "mawe":             "expression of surprise, shock, admiration",
    "hai":              "expression of surprise",
    "awe":              "expression of surprise, shock",
    "bambi":            "expression of sympathy, appreciation",
    "ai bambi":         "expression of sympathy",
    "ee":               "expression of surprise, admiration",
    "haakiri":          "expression of satisfaction",
    "cucu":             "expression of surprise",
    "mpora":            "expression of sympathy",
    "leero":            "expression of surprise, anger, fear, pleasure",
    "nyaburaiwe":       "expression of appealing, pity for oneself",
    "nyaaburanyowe":    "expression of pity, sadness",
    "mahano":           "expression of surprise, shock, disappointment",
    "caali":            "expression of kindness, appealing, admiration",
    "ndayaawe":         "appealing to someone or swearing by mother's clan",
    "nyaaburoogu":      "expression showing pity or admiration",
    "boojo":            "expression of admiration, pity, appeal, disappointment or pain",
    "mara boojo":       "expression of appealing",
    "ego":              "expression of assurance, satisfaction",
    "nukwo":            "expression of assurance, satisfaction, dissatisfaction",
    "manyeki":          "expression of doubt",
    "nga":              "expression of surprise, doubt, negation",
    "busa":             "expression of negation",
    "nangwa":           "expression of surprise, doubt, negation",
    "taata we":         "expression of surprise, shock, pity",
    "Ruhanga wange":    "my God — expression of surprise, shock, displeasure",
    "Weza":             "indeed",
    "Weebale":          "thank you",
    "hee":              "expression of surprise, shock",
    "gamba":            "expression of surprise",
    "dahira":           "expression of surprise, disbelief — literally 'swear!'",
    "ka mahano":        "expression of surprise, shock, disappointment",
    "ka kibi":          "expression of surprise, pity — literally 'it is bad!'",
    "bbaasi":           "enough!",
    "kooboine":         "expression of pity",
}


# ─────────────────────────────────────────────────────────────────────────────
# IDIOMATIC EXPRESSIONS (Ebidikizo)
# ─────────────────────────────────────────────────────────────────────────────

IDIOMS = {
    "kuburorra mu rwigi":           "leaving very early in the morning",
    "baroleriire ha liiso":         "watching over a dying person",
    "kucweke nteho ekiti":          "running as fast as possible",
    "kurubata atakincwa":           "walking hurriedly and excitedly",
    "kugenda obutarora nyuma":      "walking very fast and in a concentrated manner",
    "omutima guli enyuma":          "dissatisfied, worried",
    "omutima guramaire":            "dissatisfied, worried",
    "omutima gwezire":              "satisfied, contented",
    "amaiso kugahanga enkiro":      "waiting for somebody/something with anxiety",
    "kukwata ogwa timbabaine":      "disappear quietly",
    "kwija naamaga":                "arrive in panic and anxiety",
    "kutarorwa izooba":             "too beautiful to be exposed",
    "kuteera akahuno":              "effect of great surprise",
    "garama nkwigate":              "talk carelessly",
    "amaiso ga kimpenkirye":        "shamelessness",
    "maguru nkakwimaki":            "as fast as possible",
    "kuseka ekihiinihiini":         "laughing with great happiness",
}


# ─────────────────────────────────────────────────────────────────────────────
# PROVERBS (Enfumo)
# ─────────────────────────────────────────────────────────────────────────────

PROVERBS = [
    "Ababiri bagamba kamu, abasatu basatura",
    "Amagezi macande bakaranga nibanena",
    "Amazima obu'gaija, ebisuba biruka",
    "Buli kasozi nengo yako",
    "Ekigambo ky'omukuru mukaro, obw'ijuka onenaho",
    "Ekibi tikibura akireeta",
    "Enjara etemesa emigimba ebiri",
    "Mpora, mpora, ekahikya omunyongorozi haiziba",
    "Kamu kamu nugwo muganda",
    "Omutima guli enyuma",
    "Amaizi tigebwa owabugo mbeho",
    "Engaro ibiri kunaabisa ngana",
]

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_grammar_context() -> str:
    """Concise grammar context string for use in chat/translation prompts."""
    return (
        "Runyoro-Rutooro Grammar Rules:\n"
        "1. R/L Rule: R is dominant. L only before/after 'e' or 'i' vowels.\n"
        "2. Verb infinitives start with 'oku-' (e.g. okugenda = to go).\n"
        "3. Noun classes: omu-/aba- (people), en-/em- (animals/things), "
        "ama- (cl.6 plurals), obu- (abstract), oku- (infinitives/body parts).\n"
        "4. Tense markers: ni- (present imperfect), ka- (past), ra-/raa- (future), "
        "-ire/-ere (perfect).\n"
        "5. Subject prefixes: n- (I), o- (you sg), a- (he/she), tu- (we), "
        "mu- (you pl), ba- (they).\n"
        "6. Adjectives and numerals agree with noun class via concordial prefixes.\n"
        "7. Comparison uses okusinga (to surpass): asinga omurungi = she is better.\n"
        "8. Negation: ti- prefix on verb, e.g. tinigenda = is not going.\n"
        "9. Apostrophe marks swallowed initial vowel in fast speech: n'ente, z'ente.\n"
        "10. Long vowels written double: aa, ee, ii, oo, uu.\n"
    )


def lookup_interjection(word: str) -> str | None:
    return INTERJECTIONS.get(word.lower().strip())


def lookup_idiom(phrase: str) -> str | None:
    return IDIOMS.get(phrase.lower().strip())


def is_verb_infinitive(word: str) -> bool:
    """Return True if word looks like a Runyoro-Rutooro verb infinitive."""
    w = word.lower().strip()
    return w.startswith("oku") or w.startswith("okw")


def get_plural(singular: str) -> str | None:
    """Return known plural for a class 11 noun, or None."""
    return PLURAL_SOUND_CHANGES.get(singular.lower().strip())


def detect_noun_class_from_prefix(word: str) -> list:
    """Wrapper around get_noun_class for external use."""
    return get_noun_class(word)


def number_to_runyoro(n: int) -> str | None:
    """Return Runyoro-Rutooro word for a number if known."""
    return NUMBERS.get(n)
