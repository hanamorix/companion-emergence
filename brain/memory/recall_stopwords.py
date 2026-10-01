"""The recall token selector's stopword set.

Lives here, below the chat layer, so the known-names admission filter
(``brain.memory.known_names``, S70) can use it without importing
``brain.chat``. ``brain.chat.prompt`` keeps ``_RECALL_STOPWORDS`` as an alias
of :data:`RECALL_STOPWORDS`, so the selector and its tests are unchanged.
"""

from __future__ import annotations

# Conservative closed-class English function words + common discourse
# interjections/fillers — deliberately EXCLUDES content words ("issue",
# "first", "quick", "seems", "memory", "trigger", "signal", "logger" etc.),
# which are demoted by salience ordering, not filtered outright. English-
# specific (documented limitation). Static frozenset: no NLTK/sklearn
# dependency for a word list.
RECALL_STOPWORDS: frozenset[str] = frozenset(
    {
        # articles
        "a", "an", "the",
        # pronouns / determiners
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
        "this", "that", "these", "those",
        "who", "whom", "whose", "which", "what",
        "whoever", "whatever", "whichever",
        "any", "some", "all", "both", "each", "either", "neither",
        "every", "other", "another", "such", "own", "same", "only", "none",
        # auxiliaries / modals
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having",
        "do", "does", "did", "doing",
        "will", "would", "shall", "should", "can", "could", "may", "might", "must",
        # prepositions
        "about", "above", "across", "after", "against", "along", "among",
        "around", "at", "before", "behind", "below", "beside", "between",
        "by", "down", "during", "except", "for", "from", "in", "into",
        "near", "of", "off", "on", "out", "over", "since", "through", "to",
        "towards", "under", "until", "up", "upon", "with", "within", "without",
        # conjunctions
        "and", "but", "or", "nor", "so", "yet", "because", "although",
        "though", "while", "if", "unless", "whether", "than", "as",
        # common discourse interjections / fillers
        "alrighty", "okay", "ok", "yeah", "yep", "nope", "hmm", "anyway",
        "gonna", "wanna", "oh", "hey", "yes", "no", "alright", "hi", "hello",
    }
)
