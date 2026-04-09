"""
Runyoro-Rutooro chat assistant.
Priority order:
  1. Translation requests  (highest priority)
  2. Word lookup / meaning
  3. Exact phrase table matches
  4. Translate user Lunyoro → understand → reply in Lunyoro
  5. Contextual fallback
"""
import re
from typing import List, Dict

# ── hardcoded conversational replies (Lunyoro in, Lunyoro out) ───────────────
# Only simple greetings/farewells — NOT general words that appear in sentences
_GREETINGS = [
    (r"^oraire\s+ota[\?!.]?$",          "Nairire bulungi, webare! Wowe oraire ota?"),
    (r"^osiibire\s+ota[\?!.]?$",         "Nsiibire bulungi, webare! Wowe osiibire ota?"),
    (r"^osibye\s+ota[\?!.]?$",           "Nsibye bulungi, webare! Wowe osibye ota?"),
    (r"^(oli|uli)\s+ota[\?!.]?$",        "Ndi bulungi, webare! Wowe oli ota?"),
    (r"^(oli|uli)\s+bulungi[\?!.]?$",    "Yego, ndi bulungi! Webare okubuuza."),
    (r"^mirembe[\?!.]?$",                "Mirembe nawe!"),
    (r"^(hello|hi|hey)[\?!.]?$",         "Oraire ota! Nkuyambe ata?"),
    (r"^webare\s+nyo[\?!.]?$",           "Kya bwangu! Nkusanyukira okukuyamba."),
    (r"^webare[\?!.]?$",                 "Kya bwangu!"),
    (r"^(ogende|wendaho)\s+bulungi[\?!.]?$", "Ogende bulungi! Ruhanga akuhe omugisha."),
    (r"^turabonana[\?!.]?$",             "Yego, turabonana! Ogende bulungi."),
    (r"^(yego|eego)[\?!.]?$",            "Yego, nategeera!"),
    (r"^(nedda|nangwa|oya)[\?!.]?$",     "Kya bwangu, tihariho kizibu!"),
    (r"^(okay|ok|alright)[\?!.]?$",      "Kya bwangu!"),
    (r"^(sorry|mbabarira)[\?!.]?$",      "Tihariho kizibu! Nkuyamba ata?"),
    (r"^wiitwa\s+(ata|nata)[\?!.]?$",    "Nziitwa Omuyambi wa Runyoro-Rutooro. Ndi hano okukuyamba!"),
    (r"^(oli|uli)\s+(ani|nani)[\?!.]?$", "Ndi omuyambi w'olulimi lwa Runyoro-Rutooro."),
]

# ── intent: translate English phrase to Lunyoro ───────────────────────────────
_RE_TO_LUN = re.compile(
    r"how\s+(do|can)\s+(i|you)\s+say\s+(.+?)\s+in\s+(runyoro|rutooro|lunyoro)"
    r"|translate\s+[\"']?(.+?)[\"']?\s+(to|in(to)?)\s+(runyoro|rutooro|lunyoro)"
    r"|hindura\s+[\"']?(.+?)[\"']?\s+omu\s+(runyoro|rutooro)",
    re.IGNORECASE,
)

# ── intent: word meaning lookup ───────────────────────────────────────────────
_RE_LOOKUP = re.compile(
    r"(what\s+does|what\s+is|meaning\s+of|define|obusobanuro\s+bwa?|"
    r"kiki\s+ekigambo\s+kya?|ekigambo\s+kya?|bisobanura\s+ngu)\s+[\"']?(.+?)[\"']?"
    r"(\s+(mean|in\s+english|omu\s+english|bisobanura))?[\?!.]?$",
    re.IGNORECASE,
)

# ── intent: general question (ends with ?) ────────────────────────────────────
_RE_QUESTION = re.compile(r"\?$")


def generate_reply(message: str, history: List[Dict[str, str]] | None = None) -> Dict:
    from translate import translate, translate_to_english, lookup_word

    msg = message.strip()
    msg_lower = msg.lower().strip()

    # ── 1. Translation request: "how do you say X in Runyoro" ────────────────
    m = _RE_TO_LUN.search(msg)
    if m:
        # extract the phrase from whichever group matched
        phrase = (m.group(3) or m.group(5) or m.group(9) or "").strip().strip("'\"")
        if phrase:
            res = translate(phrase)
            translation = res.get("translation")
            if translation:
                return {"reply": f'Omu Runyoro-Rutooro, "{phrase}" bigambwa ngu: "{translation}"'}
        return {"reply": 'Wandika ekigambo mu chikomo nkugarukemu. Ekyokulabirako: "How do you say \'water\' in Runyoro?"'}

    # ── 2. Word lookup / meaning ──────────────────────────────────────────────
    m = _RE_LOOKUP.search(msg)
    if m:
        phrase = (m.group(2) or "").strip().strip("'\"")
        if phrase and len(phrase) < 60:
            direction = "lun→en" if _looks_like_lunyoro(phrase) else "en→lun"
            results = lookup_word(phrase, direction)
            if results:
                return {"reply": _format_lookup(phrase, results, direction)}
            # fallback to direct translation
            if direction == "lun→en":
                res = translate_to_english(phrase)
            else:
                res = translate(phrase)
            t = res.get("translation")
            if t:
                return {"reply": f'"{phrase}" bisobanura ngu: "{t}"'}
            return {"reply": f'Sisobola okusanga obusobanuro bwa "{phrase}". Gezaako ekigambo ekindi.'}

    # ── 3. Exact greeting/farewell matches (short messages only) ─────────────
    if len(msg_lower) < 50:
        for pattern, reply in _GREETINGS:
            if re.match(pattern, msg_lower, re.IGNORECASE):
                return {"reply": reply}

    # ── 4. Translate Lunyoro input → understand → reply in Lunyoro ───────────
    if _looks_like_lunyoro(msg):
        understood = ""
        try:
            res = translate_to_english(msg)
            understood = res.get("translation") or ""
        except Exception:
            pass

        if understood:
            reply = _intent_to_lunyoro(understood)
            if reply:
                return {"reply": reply}

        # MT gave output but intent didn't match — give contextual fallback
        reply = _lunyoro_keyword_reply(msg_lower)
        if reply:
            return {"reply": reply}

        return {"reply": "Ntegeera! Gezaako okugamba bundi, nkugarukemu."}

    # ── 5. English input — translate to Lunyoro directly ─────────────────────
    if not _looks_like_lunyoro(msg):
        try:
            res = translate(msg)
            t = res.get("translation")
            conf = res.get("confidence", 0)
            if t and conf > 0.35:
                return {"reply": t}
        except Exception:
            pass

    # ── 6. Fallback ───────────────────────────────────────────────────────────
    fallbacks = [
        "Ntegeera! Nkuyamba omu kuhindura amagambo n'okubuuza ebibuuzo by'olulimi lwa Runyoro-Rutooro.",
        "Yego, ndi hano! Buuza ekibuuzo kyawe omu Runyoro-Rutooro.",
        "Nkusanyukira okukuyamba! Wandika ekibuuzo kyawe nkugarukemu.",
        "Nkugarukemu! Buuza ekibuuzo kyawe omu Runyoro-Rutooro.",
    ]
    return {"reply": fallbacks[len(msg) % len(fallbacks)]}


# ── helpers ───────────────────────────────────────────────────────────────────

def _intent_to_lunyoro(en: str) -> str:
    """
    Map MT-translated English to a Lunyoro reply.
    Patterns match what the lun2en MarianMT model actually produces.
    """
    low = en.lower()
    low = re.sub(r"^\[general\]\s*", "", low)  # strip [GENERAL] prefix

    # ── Greetings / wellbeing ─────────────────────────────────────────────────
    if re.search(r"how are you|how do you feel|are you (well|okay|fine|good)", low):
        return "Nsiibire bulungi, webare okubuuza! Wowe osiibire ota?"
    if re.search(r"good (morning|day|afternoon)", low):
        return "Oraire bulungi! Osiibire ota?"
    if re.search(r"good (evening|night)", low):
        return "Osibye bulungi! Otuire ota?"
    if re.search(r"(what is|what'?s) your name|who are you", low):
        return "Nziitwa Omuyambi wa Runyoro-Rutooro. Ndi hano okukuyamba!"
    if re.search(r"i am (here|present|available|ready)", low):
        return "Nkusanyukira! Ndi hano nawe. Buuza ekibuuzo kyawe."
    if re.search(r"i (am helping|help|helped).+(you|learn|english|language)", low):
        return "Webare! Nkusanyukira okukuyamba. Buuza ekibuuzo kyawe."
    if re.search(r"thank(s| you)|grateful|appreciate", low):
        return "Kya bwangu! Nkusanyukira okukuyamba."
    if re.search(r"goodbye|bye|go well|farewell|see you|until next", low):
        return "Ogende bulungi! Ruhanga akuhe omugisha. Turabonana!"
    if re.search(r"\byes\b|correct|right|true|exactly|indeed", low):
        return "Yego, nategeera bulungi!"
    if re.search(r"\bno\b|not |wrong|incorrect|never", low):
        return "Kya bwangu, nategeera. Nkugarukemu!"
    if re.search(r"sorry|apologize|excuse|forgive", low):
        return "Tihariho kizibu! Nkuyamba ata?"
    if re.search(r"i (am|feel) (good|well|fine|happy|great|okay)", low):
        return "Nkusanyukira okuwulira bulungi! Nkuyamba ata?"
    if re.search(r"i (am|feel) (bad|sick|sad|tired|unwell|not well)", low):
        return "Nkusaasira! Ruhanga akuhe obuzima. Nkuyamba ata?"

    # ── Tell / explain / talk about (matches "I told you about X" MT output) ──
    if re.search(r"(tell|told|say|said|speak|talk|explain|describe|inform).+(runyoro|rutooro|bunyoro|language|culture|tradition)", low):
        return ("Runyoro-Rutooro ni olulimi lwa Bantu olukoolebwa omu Bunyoro-Kitara, Uganda. "
                "Olukoolebwa abantu ba Banyoro na Batooro. "
                "Empaako ni mazina g'okusiima — ekimu omu ebisanfu by'obusanfu bwa Bunyoro. "
                "Nkuyamba okwiga olulimi olu!")
    if re.search(r"(tell|told|say|said|speak|talk|explain|describe).+(grammar|verb|noun|tense|prefix|class)", low):
        return ("Omu Runyoro-Rutooro, ebigambo by'okukora bitandika na 'oku-': "
                "okukora (to work), okurya (to eat), okwenda (to go). "
                "Amazina gali n'ebika: omu-/aba- (abantu), eki-/ebi- (ebintu), aka-/aga- (ebintu binini).")
    if re.search(r"(tell|told|say|said|speak|talk|explain|describe).+(number|count|counting)", low):
        return "Okubarura omu Runyoro: 1=emu, 2=ibiri, 3=isatu, 4=ina, 5=itaano, 6=mukaaga, 7=musanju, 8=munaana, 9=mwenda, 10=ikumi."
    if re.search(r"(tell|told|say|said|speak|talk|explain|describe).+(food|eat|drink|cook)", low):
        return "Eky'okulya omu Runyoro: Emmere (food), Ebitooke (bananas), Obushera (porridge), Enva (vegetables), Enyama (meat)."
    if re.search(r"(tell|told|say|said|speak|talk|explain|describe).+(family|mother|father|child|wife|husband)", low):
        return "Amagambo g'enju: Taata (father), Maama (mother), Omwana (child), Omukazi (woman/wife), Omushaija (man/husband), Muganda (sibling)."
    if re.search(r"(tell|told|say|said|speak|talk|explain|describe|inform|give).+(about|concerning|regarding|on|seven|times|more)", low):
        return "Yego, nkugambira! Buuza ekibuuzo kyawe nkugarukemu."

    # ── Help / assist ─────────────────────────────────────────────────────────
    if re.search(r"help (me|you|us)|assist|support|i need help", low):
        return "Yego, nkuyamba! Buuza ekibuuzo kyawe nkugarukemu."
    if re.search(r"(help|assist).+(learn|study|practice|understand)", low):
        return ("Bulungi nnyo okwiga Runyoro-Rutooro! "
                "Tandika n'okwega okubaza: 'Osiibire ota?' (How are you?), "
                "'Webare' (Thank you), 'Ogende bulungi' (Go well).")
    if re.search(r"(help|assist).+(translate|translation|word|meaning)", low):
        return "Nkuyamba okuhindura! Wandika ekigambo nkugarukemu."

    # ── Questions / asking ────────────────────────────────────────────────────
    if re.search(r"i (ask|asked|have a question|want to ask)", low):
        return "Yego, buuza ekibuuzo kyawe nkugarukemu!"
    if re.search(r"what (about|is|are|does).+(woman|man|child|person|people)", low):
        return "Nkuyamba okumanya ebigambo! Wandika ekigambo nkugarukemu."
    if re.search(r"(what|how|when|where|why|who).+(runyoro|rutooro|word|mean|say|translate)", low):
        return "Ekibuuzo kyawe kirungi! Nkugarukemu. Buuza bundi nkugarukemu."

    # ── Learning ──────────────────────────────────────────────────────────────
    if re.search(r"(learn|study|practice|teach|understand).+(runyoro|rutooro|language|lunyoro|english)", low):
        return ("Bulungi nnyo okwiga Runyoro-Rutooro! "
                "Tandika n'okwega okubaza: 'Osiibire ota?' (How are you?), "
                "'Webare' (Thank you), 'Ogende bulungi' (Go well).")
    if re.search(r"learn|study|practice|teach|understand", low):
        return "Nkuyamba okwiga! Buuza ekibuuzo kyawe omu Runyoro-Rutooro."

    # ── Translate requests ────────────────────────────────────────────────────
    if re.search(r"translate|translation|how.+say|what.+word", low):
        return "Nkuyamba okuhindura! Wandika ekigambo nkugarukemu."

    # ── Want / need ───────────────────────────────────────────────────────────
    if re.search(r"i (want|need|would like|wish|desire)", low):
        return "Ntegeera! Nkuyamba okufuna ekyotagisa. Buuza ekibuuzo kyawe."

    # ── Please ────────────────────────────────────────────────────────────────
    if re.search(r"please|kindly|i beg", low):
        return "Yego, nkuyamba! Buuza ekibuuzo kyawe."

    # ── Understand ────────────────────────────────────────────────────────────
    if re.search(r"(i |do you )?(understand|understood|clear|i see|got it)", low):
        return "Bulungi! Nkusanyukira. Buuza ekibuuzo ekindi."

    # ── What can you do ───────────────────────────────────────────────────────
    if re.search(r"what (can you|do you) do|how (can you|do you) help|your (job|work|purpose)", low):
        return ("Nkuyamba omu:\n"
                "• Kuhindura amagambo (English ↔ Runyoro-Rutooro)\n"
                "• Okumanya obusobanuro bw'ebigambo\n"
                "• Okubuuza ebibuuzo by'olulimi n'obusanfu")

    # ── Numbers ───────────────────────────────────────────────────────────────
    if re.search(r"number|count|one|two|three|four|five", low):
        return "Okubarura omu Runyoro: 1=emu, 2=ibiri, 3=isatu, 4=ina, 5=itaano, 6=mukaaga, 7=musanju, 8=munaana, 9=mwenda, 10=ikumi."

    # ── Family ────────────────────────────────────────────────────────────────
    if re.search(r"family|mother|father|child|wife|husband|sibling|brother|sister", low):
        return "Amagambo g'enju: Taata (father), Maama (mother), Omwana (child), Omukazi (woman/wife), Omushaija (man/husband), Muganda (sibling)."

    # ── Food ──────────────────────────────────────────────────────────────────
    if re.search(r"food|eat|drink|cook|meal|hunger", low):
        return "Eky'okulya omu Runyoro: Emmere (food), Ebitooke (bananas), Obushera (porridge), Enva (vegetables), Enyama (meat)."

    # ── Grammar ───────────────────────────────────────────────────────────────
    if re.search(r"verb|grammar|noun|tense|prefix|suffix|class", low):
        return ("Omu Runyoro-Rutooro, ebigambo by'okukora bitandika na 'oku-': "
                "okukora (to work), okurya (to eat), okwenda (to go). "
                "Amazina gali n'ebika: omu-/aba- (abantu), eki-/ebi- (ebintu).")

    # ── Culture ───────────────────────────────────────────────────────────────
    if re.search(r"culture|tradition|empaako|bunyoro|kingdom|history", low):
        return ("Empaako ni mazina g'okusiima agahabwa abantu omu Bunyoro-Kitara. "
                "Ni mazina 12 agakoreshebwa omu Batooro, Banyoro, n'abandi. "
                "Obunyoro-Kitara bwali obukama obusingaho omu Africa ya Mpaka.")

    # ── Time / date ───────────────────────────────────────────────────────────
    if re.search(r"time|day|today|tomorrow|yesterday|week|month|year", low):
        return "Sirina obwire bw'aha, naye nkuyamba omu Runyoro-Rutooro! Buuza ekibuuzo kyawe."

    # ── Greetings catch-all ───────────────────────────────────────────────────
    if re.search(r"good|morning|evening|night|afternoon|hello|hi|greet", low):
        return "Oraire ota! Nkuyambe ata?"

    return ""


def _lunyoro_keyword_reply(msg_lower: str) -> str:
    """Keyword fallback when MT output doesn't match any intent."""
    if any(w in msg_lower for w in ["runyoro", "rutooro", "lunyoro", "bunyoro"]):
        return ("Runyoro-Rutooro ni olulimi lwa Bantu olukoolebwa omu Bunyoro-Kitara, Uganda. "
                "Nkuyamba okwiga olulimi olu! Buuza ekibuuzo kyawe.")
    if any(w in msg_lower for w in ["hindura", "kuhindura"]):
        return "Nkuyamba okuhindura! Wandika ekigambo nkugarukemu."
    if any(w in msg_lower for w in ["ekigambo", "obusobanuro", "bisobanura"]):
        return "Nkuyamba okumanya ebigambo! Wandika ekigambo nkugarukemu."
    if any(w in msg_lower for w in ["okwiga", "kwiga"]):
        return ("Bulungi nnyo okwiga Runyoro-Rutooro! "
                "Tandika n'okwega okubaza: 'Osiibire ota?' (How are you?), "
                "'Webare' (Thank you), 'Ogende bulungi' (Go well).")
    if any(w in msg_lower for w in ["nkuyambe", "nkuyamba", "okukuyamba"]):
        return "Yego, nkuyamba! Buuza ekibuuzo kyawe nkugarukemu."
    if any(w in msg_lower for w in ["nkugambire", "ngambire", "gambira"]):
        return "Yego, nkugambira! Buuza ekibuuzo kyawe nkugarukemu."
    if any(w in msg_lower for w in ["obusanfu", "obusanfu"]):
        return ("Empaako ni mazina g'okusiima agahabwa abantu omu Bunyoro-Kitara. "
                "Obunyoro-Kitara bwali obukama obusingaho omu Africa ya Mpaka.")
    return ""


def _looks_like_lunyoro(text: str) -> bool:
    words = text.lower().split()
    if not words:
        return False
    prefixes = ("om", "ob", "ok", "ek", "eb", "ab", "en", "em", "oru", "ama",
                "obu", "otu", "eri", "aga", "ege", "oku", "okw",
                "nk", "nd", "ns", "nt", "ng", "mb", "mp", "nz", "ny",
                "bw", "by", "ky", "gy", "tw", "rw", "kw")
    hits = sum(1 for w in words if any(w.startswith(p) for p in prefixes))
    # match if at least 1 word looks Lunyoro in short messages, or >30% in longer ones
    threshold = 1 if len(words) <= 3 else max(1, len(words) // 3)
    return hits >= threshold


def _format_lookup(word: str, results: list, direction: str) -> str:
    lines = [f'Ebigambo bya "{word}":\n']
    for r in results[:3]:
        w = r.get("word", "")
        def_en = r.get("definitionEnglish", "")
        def_nat = r.get("definitionNative", "")
        pos = r.get("pos", "")
        ex = r.get("exampleSentence1", "")
        ex_en = r.get("exampleSentence1English", "")
        entry = f"• {w}"
        if pos:
            entry += f" ({pos})"
        if def_en:
            entry += f" — {def_en}"
        if def_nat:
            entry += f" / {def_nat}"
        lines.append(entry)
        if ex:
            lines.append(f"  Ekyokulabirako: {ex}" + (f" ({ex_en})" if ex_en else ""))
    return "\n".join(lines).strip()
