"""Shared deterministic facts and official resources for the COMSATS
chatbot (used by both the Streamlit UI and the CLI RAG script).

Deterministic facts are list/count-type facts that vector similarity
retrieval answers unreliably, so they are injected (or returned
verbatim) by keyword match instead of depending on retrieval.
Keeping them in one module avoids the two entry points drifting apart.
"""

import difflib
import re

DETERMINISTIC_FACTS = {
    "campus": (
        "COMSATS University Islamabad (CUI) has 7 physical campuses: "
        "Islamabad (Principal Seat, established 1998), Abbottabad (2001), "
        "Wah (2001), Lahore (2002), Attock (2004), Sahiwal (2006), and "
        "Vehari (2008), plus a Virtual Campus (2012) for online/distance learning."
    ),
    "department": (
        "COMSATS University Islamabad (Islamabad Campus) has 13 departments: "
        "Architecture, Art & Design, Computer Science, Electrical Engineering, "
        "Management Sciences, Mathematics, Physics, Chemistry, Bio Sciences, "
        "Humanities, Meteorology, Economics, and Health Informatics."
    ),
    "program": (
        "COMSATS University Islamabad (Islamabad Campus) offers undergraduate (BS), "
        "graduate (MS), and doctoral (PhD) programs spread across its 13 departments: "
        "Architecture, Art & Design, Computer Science, Electrical Engineering, "
        "Management Sciences, Mathematics, Physics, Chemistry, Bio Sciences, "
        "Humanities, Meteorology, Economics, and Health Informatics. The exact "
        "programs offered vary by department — ask about a specific department "
        "(e.g. 'What programs does Computer Science offer?') for details."
    ),
    "scholarship": (
        "For scholarships, please visit: https://ww2.comsats.edu.pk/scholarships/"
    ),
}


def get_deterministic_facts(query: str) -> str:
    """Return the deterministic facts whose keyword matches the query
    (exact/prefix match first, then a typo-tolerant fuzzy fallback)."""
    q = query.lower()
    words = re.findall(r"[a-z]+", q)
    matched = []
    for kw, fact in DETERMINISTIC_FACTS.items():
        if re.search(rf"\b{kw}", q):
            matched.append(fact)
            continue
        if any(difflib.SequenceMatcher(None, kw, w).ratio() > 0.75 for w in words):
            matched.append(fact)
    return "\n\n".join(matched)


# High-value OFFICIAL resources to show when the topic matches, even
# when the user did not ask for a download/link. These are more reliable
# than scraped text (fee/admission text can mix batches and go stale).
# Each entry is a (title, url) pair.
DETERMINISTIC_RESOURCES = {
    r"\badmission\b.*\bfall\b|\bfall\b.*\badmission\b|\badmissions?\b": [
        (
            "Fall 2026 Admission Notice (Official)",
            "https://islamabad.comsats.edu.pk/NEFiles/Admission-fall-2026-isb.jpg",
        ),
    ],
}


def get_deterministic_resources(query: str) -> list:
    """Return [(title, url), ...] official resources matching the query."""
    q = query.lower()
    matched = []
    for pattern, resources in DETERMINISTIC_RESOURCES.items():
        if re.search(pattern, q):
            matched.extend(resources)
    return matched
