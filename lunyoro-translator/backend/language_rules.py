"""
Runyoro-Rutooro language rules extracted from runyorodictionary.com
Used for grammar guidance, intent detection, and chat context.
"""

# ── Empaako (Honorific Names) ─────────────────────────────────────────────────
EMPAAKO = {
    "Atwooki":  "From Atwok — shining star. Given to a child born on a night with stars.",
    "Ateenyi":  "From Ateng — beautiful. Given if parents believe the child is beautiful.",
    "Apuuli":   "From Apul/Rapuli — a very lovely girl, center of attraction and love in the family.",
    "Amooti":   "From Amot — princely, a sign of royalty. Mostly for sons and daughters of kings and chiefs.",
    "Akiiki":   "From Ochii/Achii — the one who follows twins. Mostly for firstborns who bear responsibility for siblings.",
    "Adyeeri":  "From adyee/odee — parents had failed to get a child and only got one after spiritual intervention. Fortunate/truth.",
    "Acaali":   "From Ochal — replica. Given to a child who resembled someone in the family or ancestors.",
    "Abaala":   "From Obal/Abal — destroyer/warrior. Usually for sons of chiefs.",
    "Abbooki":  "From Obok/Abok — beloved. Child born out of strong love between parents.",
    "Araali":   "From Arali/Olal/Alal — lost. Given to the only surviving child after mother lost many children.",
    "Abwooli":  "From Abwolo — 'I lied to you'. A woman who conceives but continues having periods; child born of this circumstance.",
    "Okaali":   "From 'kal' — royalty. Only for the King in Kitara customs.",
}

# ── Interjections (Ebihunaazo) ────────────────────────────────────────────────
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

# ── Idiomatic Expressions (Ebidikizo) ─────────────────────────────────────────
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

# ── Numbers (Okubara) ─────────────────────────────────────────────────────────
NUMBERS = {
    1: "emu", 2: "ibiri", 3: "isatu", 4: "ina", 5: "itano",
    6: "mukaaga", 7: "musanju", 8: "munaana", 9: "mwenda", 10: "ikumi",
    11: "ikumi nemu", 20: "abiri", 30: "asatu", 40: "ana", 50: "atano",
    60: "nkaaga", 70: "nsanju", 80: "kinaana", 90: "kyenda",
    100: "kikumi", 200: "bibiri", 1000: "rukumi",
    1_000_000: "akakaikuru", 1_000_000_000: "akasirira",
}

# ── Proverbs (Enfumo) — sample ────────────────────────────────────────────────
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


def get_grammar_context() -> str:
    """Return a concise grammar context string for use in chat responses."""
    return (
        "Runyoro-Rutooro Grammar Rules:\n"
        "1. R/L Rule: R is dominant. L only before/after 'e' or 'i' vowels.\n"
        "2. Verb infinitives start with 'oku-' (e.g. okugenda = to go).\n"
        "3. Noun classes: om-/ab- (people), en-/em- (animals/things), ama- (plurals), obu- (abstract).\n"
        "4. Tense markers: ni- (present), ka- (past), ra- (future).\n"
        "5. Subject prefixes: n- (I), o- (you), a- (he/she), tu- (we), mu- (you pl), ba- (they).\n"
    )


def lookup_interjection(word: str) -> str | None:
    """Return the meaning of a Lunyoro interjection if known."""
    return INTERJECTIONS.get(word.lower().strip())


def lookup_idiom(phrase: str) -> str | None:
    """Return the meaning of a Lunyoro idiomatic expression if known."""
    return IDIOMS.get(phrase.lower().strip())
