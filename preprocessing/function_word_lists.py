"""
Explicit function-word lists for the 9 LIWC-aligned families, per language.
Surface forms are stored in lowercase; French elision contractions (l', d', j', …)
are represented by their pre-apostrophe fragment (l, d, j, …) because the
tokenizer splits on apostrophes.

Tokenization note: the regex tokenizer splits on apostrophes, so "don't" →
["don", "t"]. The negation suffix tracked here is therefore "t" (not "nt").

Words may appear in more than one family (e.g. "never" in both adverb and
negate) just as in LIWC-22.
"""

LISTS: dict[str, dict[str, frozenset]] = {
    # ── English ──────────────────────────────────────────────────────────────
    "en": {
        "article": frozenset({
            "a", "an", "the",
        }),

        "auxverb": frozenset({
            # be
            "be", "am", "is", "are", "was", "were", "been", "being",
            # have
            "have", "has", "had", "having",
            # do
            "do", "does", "did",
            # modals
            "will", "would", "shall", "should",
            "may", "might", "must", "can", "could",
            "need", "dare", "ought",
            # semi-modals
            "used",        # "used to"
            # contracted auxiliaries — apostrophe splits "I've" → ["i","ve"] etc.
            "ve",          # 've (have)
            "ll",          # 'll (will/shall)
            "re",          # 're (are)
            "d",           # 'd (would/had)
            "m",           # 'm (am)
            "s",           # 's (is/has) — also possessive, unavoidable overlap
        }),

        "conj": frozenset({
            # coordinating (FANBOYS minus ambiguous "for")
            "and", "but", "or", "nor", "yet", "so",
            # subordinating
            "although", "though",
            "because", "since", "unless", "until", "while", "whereas",
            "if", "whether", "when", "whenever", "where", "wherever",
            "after", "before", "as", "once", "than", "that",
            "provided", "lest", "albeit",
        }),

        "adverb": frozenset({
            # degree / intensifiers
            "very", "quite", "rather", "pretty", "fairly",
            "extremely", "highly", "incredibly", "terribly",
            # affirmation / certainty
            "actually", "certainly", "clearly", "definitely", "indeed",
            "obviously", "really", "truly", "truly", "undoubtedly",
            # probability / hedging
            "apparently", "maybe", "perhaps", "possibly", "probably",
            # frequency
            "always", "constantly", "ever", "frequently", "generally",
            "never", "occasionally", "often", "rarely", "seldom",
            "sometimes", "usually",
            # time
            "again", "already", "before", "eventually", "finally",
            "immediately", "just", "lately", "now", "recently",
            "simultaneously", "slowly", "so", "soon", "still",
            "suddenly", "then", "yet",
            # manner / process
            "almost", "also", "altogether", "anyway", "approximately",
            "basically", "briefly", "carefully", "closely", "completely",
            "directly", "easily", "especially", "essentially", "exactly",
            "fully", "gradually", "hard", "initially", "largely", "mainly",
            "merely", "mostly", "much", "nearly", "only", "otherwise",
            "particularly", "primarily", "quickly", "simply", "slightly",
            "specifically", "together", "too",
            # place
            "anywhere", "away", "back", "down", "everywhere", "here",
            "nowhere", "somewhere", "there", "up",
            # discourse / connector
            "accordingly", "furthermore", "hence", "however", "instead",
            "meanwhile", "moreover", "nevertheless", "nonetheless",
            "therefore", "thus", "well",
        }),

        "ppron": frozenset({
            "i", "me", "my", "mine", "myself",
            "we", "us", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "they", "them", "their", "theirs", "themselves",
        }),

        "ipron": frozenset({
            "it", "its", "itself",
            "this", "that", "these", "those",
            # indefinite
            "anyone", "anybody", "anything",
            "everyone", "everybody", "everything",
            "nobody", "nothing",
            "somebody", "someone", "something",
            "one",
            # distributive / negative
            "each", "either", "neither",
            # relative / interrogative
            "what", "whatever", "which", "whichever",
            "who", "whoever", "whom", "whose",
        }),

        "negate": frozenset({
            "no", "not", "none", "nor", "never",
            "neither", "nobody", "nothing", "nowhere",
            # n't split by apostrophe tokeniser → suffix "t"
            "t",
            "hardly", "scarcely", "barely",
        }),

        "prep": frozenset({
            "about", "above", "across", "after", "against", "along",
            "among", "amongst", "around", "at", "before", "behind",
            "below", "beneath", "beside", "besides", "between", "beyond",
            "by", "concerning", "despite", "down", "during", "except",
            "following", "for", "from", "in", "inside", "instead",
            "into", "like", "near", "of", "off", "on", "onto", "out",
            "outside", "over", "past", "regarding", "since", "through",
            "throughout", "till", "to", "toward", "towards", "under",
            "underneath", "unlike", "until", "up", "upon", "via",
            "with", "within", "without",
        }),

        "quant": frozenset({
            "all", "another", "any", "both", "each", "enough",
            "every", "few", "less", "little", "many", "more",
            "most", "much", "none", "other", "several", "some",
        }),

        "subord": frozenset({
            # temporal
            "when", "whenever", "while", "after", "before",
            "until", "once", "since", "as",
            # causal
            "because",
            # conditional
            "if", "unless", "whether", "provided",
            # concessive
            "although", "though", "whereas", "albeit",
            # purpose / result
            "lest", "that",
            # comparative
            "than",
        }),
    },

    # ── French ───────────────────────────────────────────────────────────────
    # Elision fragments (l, d, j, m, t, s, n, qu, c) are included so that
    # "l'homme", "j'ai", "n'est", "qu'il" etc. are counted after
    # apostrophe-based tokenisation.
    "fr": {
        "article": frozenset({
            "le", "la", "les", "l",          # l' (elision)
            "un", "une", "des",
            "du", "au", "aux",
        }),

        "auxverb": frozenset({
            # ── être ─────────────────────────────────────────────────────
            "être",
            # présent
            "suis", "es", "est", "sommes", "êtes", "sont",
            # imparfait
            "étais", "était", "étions", "étiez", "étaient",
            # futur simple
            "serai", "seras", "sera", "serons", "serez", "seront",
            # conditionnel présent
            "serais", "serait", "serions", "seriez", "seraient",
            # subjonctif présent
            "sois", "soit", "soyons", "soyez", "soient",
            # participe passé + gérondif
            "été", "étant",

            # ── avoir ────────────────────────────────────────────────────
            "avoir",
            # présent
            "ai", "as", "a", "avons", "avez", "ont",
            # imparfait
            "avais", "avait", "avions", "aviez", "avaient",
            # futur simple
            "aurai", "auras", "aura", "aurons", "aurez", "auront",
            # conditionnel présent
            "aurais", "aurait", "aurions", "auriez", "auraient",
            # subjonctif présent
            "aie", "aies", "ait", "ayons", "ayez", "aient",
            # participe passé + gérondif
            "eu", "ayant",

            # ── aller (futur proche) ─────────────────────────────────────
            "aller",
            # présent
            "vais", "vas", "va", "allons", "allez", "vont",
            # imparfait
            "allais", "allait", "allions", "alliez", "allaient",
            # futur simple
            "irai", "iras", "ira", "irons", "irez", "iront",
            # conditionnel
            "irais", "irait", "irions", "iriez", "iraient",

            # ── pouvoir (can / may) ──────────────────────────────────────
            "pouvoir",
            # présent
            "peux", "peut", "pouvons", "pouvez", "peuvent",
            # imparfait
            "pouvais", "pouvait", "pouvions", "pouviez", "pouvaient",
            # futur simple
            "pourrai", "pourras", "pourra", "pourrons", "pourrez", "pourront",
            # conditionnel
            "pourrais", "pourrait", "pourrions", "pourriez", "pourraient",
            # subjonctif
            "puisse", "puisses", "puissions", "puissiez", "puissent",
            # participe passé + gérondif
            "pu", "pouvant",

            # ── devoir (must / have to) ──────────────────────────────────
            "devoir",
            # présent
            "dois", "doit", "devons", "devez", "doivent",
            # imparfait
            "devais", "devait", "devions", "deviez", "devaient",
            # futur simple
            "devrai", "devras", "devra", "devrons", "devrez", "devront",
            # conditionnel
            "devrais", "devrait", "devrions", "devriez", "devraient",
            # subjonctif
            "doive", "doives", "doivent",
            # participe passé + gérondif
            "dû", "devant",

            # ── vouloir (want / will) ────────────────────────────────────
            "vouloir",
            # présent
            "veux", "veut", "voulons", "voulez", "veulent",
            # imparfait
            "voulais", "voulait", "voulions", "vouliez", "voulaient",
            # futur simple
            "voudrai", "voudras", "voudra", "voudrons", "voudrez", "voudront",
            # conditionnel
            "voudrais", "voudrait", "voudrions", "voudriez", "voudraient",
            # subjonctif
            "veuille", "veuilles", "veuillent", "voulions", "vouliez",
            # participe passé + gérondif
            "voulu", "voulant",

            # ── falloir (must / be necessary) ────────────────────────────
            "falloir",
            # présent (impersonal)
            "faut",
            # imparfait
            "fallait",
            # futur simple
            "faudra",
            # conditionnel
            "faudrait",
            # subjonctif
            "faille",
            # participe passé
            "fallu",

            # ── savoir (know how to) ──────────────────────────────────────
            "savoir",
            # présent
            "sais", "sait", "savons", "savez", "savent",
            # imparfait
            "savais", "savait", "savions", "saviez", "savaient",
            # futur simple
            "saurai", "sauras", "saura", "saurons", "saurez", "sauront",
            # conditionnel
            "saurais", "saurait", "saurions", "sauriez", "sauraient",
            # subjonctif
            "sache", "saches", "sachons", "sachez", "sachent",
            # participe passé + gérondif
            "su", "sachant",

            # contracted auxiliaries (apostrophe splits "j'ai" → ["j","ai"])
            # j (j'), m (m'), t (t'), s (s'), n (n') — already in ppron/ipron
            # but their auxiliary reading is captured by the verb forms above
        }),

        "conj": frozenset({
            # coordinating (LIWC-22 FR: et, mais, ou, donc, or, ni, car)
            "et", "mais", "ou", "donc", "or", "ni", "car",
            # subordinating
            "que", "qu",                      # qu' (elision of "que")
            "quand", "comme", "si",
            "lorsque", "puisque",
            "parce",                          # parce que → ["parce","que"]
            "afin",                           # afin que → ["afin","que"]
            "tandis",                         # tandis que → ["tandis","que"]
            "quoique", "pourvu",              # pourvu que
            "bien",                           # bien que → ["bien","que"] — "bien" also adverb
            "cependant", "pourtant", "néanmoins", "toutefois",
            "sinon", "autrement",
        }),

        "adverb": frozenset({
            # degree / intensifiers
            "très", "trop", "assez", "plutôt", "vraiment",
            "beaucoup", "bien", "fort", "si",
            # frequency
            "toujours", "souvent", "parfois", "rarement", "jamais",
            # time
            "maintenant", "déjà", "encore", "bientôt", "enfin",
            "soudain", "aussitôt", "dorénavant", "désormais",
            # place
            "ici", "là", "ailleurs", "partout", "nulle",   # nulle part
            # manner / degree
            "ainsi", "alors", "aussi", "autant", "davantage",
            "ensemble", "environ", "longtemps", "même", "moins",
            "plus", "presque", "seulement", "vite", "loin", "près",
            # affirmation / negation (response words)
            "oui", "non", "si",               # "si" = yes (contradicting negation)
            # discourse
            "certes", "certainement", "évidemment", "probablement",
            "peut",                           # peut-être → ["peut","être"]
            "finalement", "simplement", "exactement", "rapidement",
            "doucement", "surtout", "notamment", "effectivement",
            "comment", "où",                  # interrogative adverbs
        }),

        "ppron": frozenset({
            "je", "j",                        # j' (elision)
            "tu",
            "il", "elle",
            "nous", "vous",
            "ils", "elles",
            "me", "m",                        # m' (elision)
            "te", "t",                        # t' (elision)
            "se", "s",                        # s' (elision)
            "lui", "leur",
            "y", "en",
            "moi", "toi", "soi",
        }),

        "ipron": frozenset({
            "ce", "c",                        # c' (elision)
            "cela", "ça", "ceci",
            "tout", "tous", "toute", "toutes",
            "chacun", "chacune",
            "quelqu",                         # quelqu'un → ["quelqu","un"]
            "rien", "personne",
            "quelque", "quelques",
            "certains", "certaines",
            "qui", "quoi",
            "lequel", "laquelle", "lesquels", "lesquelles",
            "auquel", "auxquels", "duquel", "desquels",
            "on",                             # impersonal "on" (one / we)
        }),

        "negate": frozenset({
            "ne", "n",                        # n' (elision)
            "pas", "point", "guère", "nullement",
            "jamais",                         # also adverb
            # "plus" is not listed: in "ne … plus" the negation is already caught by
            # ne/n', and on its own "plus" usually means "more" (it stays in adverb/quant)
            "rien", "personne",               # also ipron
            "ni", "non",
            "sans",                           # sans = without (also prep)
        }),

        "prep": frozenset({
            "de", "d",                        # d' (elision)
            "à",
            "en", "dans", "sur", "par",
            "pour", "avec", "sans", "sous",
            "vers", "chez", "entre", "contre",
            "depuis", "avant", "après", "pendant",
            "selon", "malgré", "hors", "parmi",
            "jusque", "auprès", "autour",
            "dès", "lors", "jusqu",           # jusqu'à → ["jusqu","à"]
        }),

        "quant": frozenset({
            "tout", "tous", "toute", "toutes",
            "plusieurs", "quelques",
            "beaucoup", "peu", "assez", "trop", "autant",
            "moins", "plus", "chaque", "davantage",
            "certain", "certains", "certaine", "certaines",
            "aucun", "aucune", "nul", "nulle",
            "quelque",                        # quelque chose → any amount
        }),

        "subord": frozenset({
            # temporal
            "quand", "lorsque",
            "pendant",                        # pendant que → ["pendant","que"]
            "avant",                          # avant que
            "après",                          # après que
            "dès",                            # dès que
            # causal
            "puisque", "parce",               # parce que → ["parce","que"]
            # conditional
            "si",
            "pourvu",                         # pourvu que
            "sinon",
            # concessive
            "bien",                           # bien que → ["bien","que"]
            "quoique",
            # purpose
            "afin",                           # afin que → ["afin","que"]
            "pour",                           # pour que
            # comparative
            "comme", "que", "qu",
            # other multi-word starters
            "tandis",                         # tandis que
        }),
    },
}
