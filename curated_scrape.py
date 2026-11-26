"""
COMSATS Islamabad — Curated Scraper (poori site crawl NAHI karta)
====================================================================
Iske bajaye ke pura site crawl kiya jaye (jisse garbage/other-campus
data mix ho jata hai), ye script sirf ek FIXED list of important,
hand-picked pages se data leti hai — har page ko ek category tag
milta hai (admissions, fees, scholarships, programs, facilities,
contacts, policies, about).

Ye COMSATS Abbottabad wale chatbot jaisa approach hai — "trusted,
curated knowledge sources" har random page nahi.

USAGE:
    pip install requests beautifulsoup4 pypdf lxml
    python curated_scrape.py

OUTPUT:
    output/all_documents.json   -> same format jo comsats_rag.py expect karta hai
"""

import hashlib
import io
import json
import logging
import os
import re
import time

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; COMSATS-RAG-Bot/1.0; "
    "+educational-project-for-chatbot-research)"
}
REQUEST_DELAY = 1.0
TIMEOUT = 15
MAX_RETRIES = 3
OUTPUT_DIR = "output"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("curated_scraper")


def fetch_with_retry(session: requests.Session, url: str):
    """Transient network errors (timeout, connection reset) aur HTTP
    429/5xx ke liye retry karta hai — warna ek chhota si glitch bhi
    us page ko permanently skip kar deta tha."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                if attempt < MAX_RETRIES:
                    wait = attempt * 2
                    log.warning(
                        f"Attempt {attempt} got {last_error} for {url}, retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
            return resp
        except requests.RequestException as e:
            last_error = e
            if attempt < MAX_RETRIES:
                wait = attempt * 2
                log.warning(
                    f"Attempt {attempt} failed for {url} ({e}), retrying in {wait}s..."
                )
                time.sleep(wait)
    log.warning(f"All {MAX_RETRIES} attempts failed for {url}: {last_error}")
    return None


# ---------------------------------------------------------------------------
# CURATED PAGE LIST — yahan edit/add/remove karo jo bhi important lage
# Har entry: (url, category)
# ---------------------------------------------------------------------------

PAGES = [
    # About / General
    ("https://islamabad.comsats.edu.pk/about-comsats.aspx", "about"),
    ("https://islamabad.comsats.edu.pk/rector-comsats.aspx", "about"),
    ("https://islamabad.comsats.edu.pk/incharge-campus.aspx", "about"),
    ("https://islamabad.comsats.edu.pk/contact-us.aspx", "contacts"),
    # Admissions
    ("https://islamabad.comsats.edu.pk/comsats-admissions.aspx", "admissions"),
    ("https://islamabad.comsats.edu.pk/prospectus.aspx", "admissions"),
    ("https://islamabad.comsats.edu.pk/fee-structure.aspx", "fees"),
    # Academics / Programs / Departments
    ("https://islamabad.comsats.edu.pk/comsats-academics.aspx", "programs"),
    ("https://islamabad.comsats.edu.pk/faculties-and-departments.aspx", "programs"),
    ("https://islamabad.comsats.edu.pk/comsats-departments.aspx", "programs"),
    ("https://ww2.comsats.edu.pk/cs", "programs"),
    ("https://ww2.comsats.edu.pk/ee", "programs"),
    ("https://ww2.comsats.edu.pk/ms", "programs"),
    ("https://ww2.comsats.edu.pk/math", "programs"),
    ("https://ww2.comsats.edu.pk/phy", "programs"),
    ("https://ww2.comsats.edu.pk/chemistry", "programs"),
    ("https://ww2.comsats.edu.pk/arch", "programs"),
    ("https://ww2.comsats.edu.pk/artdesign", "programs"),
    ("https://ww2.comsats.edu.pk/bio", "programs"),
    ("https://ww2.comsats.edu.pk/humanities", "programs"),
    ("https://ww2.comsats.edu.pk/Economics", "programs"),
    ("https://ww2.comsats.edu.pk/hi", "programs"),
    # Research
    ("https://islamabad.comsats.edu.pk/comsats-research.aspx", "research"),
    # Facilities / Campus life
    ("https://islamabad.comsats.edu.pk/campus-life.aspx", "facilities"),
    ("https://islamabad.comsats.edu.pk/junaid-zaidi-library.aspx", "facilities"),
    ("https://islamabad.comsats.edu.pk/health-care.aspx", "facilities"),
    ("https://islamabad.comsats.edu.pk/student-affairs.aspx", "facilities"),
    ("https://islamabad.comsats.edu.pk/sports-facilities.aspx", "facilities"),
    ("https://islamabad.comsats.edu.pk/comsats-cafeteria.aspx", "facilities"),
    ("https://islamabad.comsats.edu.pk/international-students.aspx", "facilities"),
    ("https://islamabad.comsats.edu.pk/halls-reservation-schedule.aspx", "facilities"),
    # Scholarships / Career
    # NOTE: "ww2.comsats.edu.pk/scholarships" (old) was a shallow landing
    # page with just "Read More" links, no real content. Ye 2 pages asli
    # scholarship criteria/details rakhte hain.
    (
        "https://ww2.comsats.edu.pk/scholarships/scholarships-and-support-programs.aspx",
        "scholarships",
    ),
    (
        "https://ww2.comsats.edu.pk/scholarships/internal-scholarships-data.aspx",
        "scholarships",
    ),
    ("https://ww2.comsats.edu.pk/cdc", "scholarships"),
    # News / Downloads
    ("https://islamabad.comsats.edu.pk/comsats-news.aspx", "news"),
    ("https://islamabad.comsats.edu.pk/downloads.aspx", "policies"),
]

# Kuch chhoti, stable, general facts jo scrape nahi ki jatin (jaan-boojh kar,
# cross-campus data-mixing se bachne ke liye) lekin chatbot ko pata hona
# chahiye — jaise campuses ke naam. Manually likha gaya, source-verified.
MANUAL_FACTS = [
    {
        "id": "manual-campuses-list",
        "url": "https://www.comsats.edu.pk/comsats-campuses.aspx",
        "type": "manual",
        "category": "about",
        "title": "COMSATS University Islamabad — Campuses",
        "content": (
            "COMSATS University Islamabad (CUI) is a multi-campus university with "
            "its Principal Seat in Islamabad. CUI has 7 physical campuses: "
            "Islamabad (Principal Seat, established 1998), Abbottabad (2001), "
            "Wah (2001), Lahore (2002), Attock (2004), Sahiwal (2006), and "
            "Vehari (2008). In addition, CUI has a Virtual Campus (established "
            "2012) offering online and distance learning programs. Each campus "
            "operates its own departments, faculty, and facilities."
        ),
    },
]

# Kuch important PDFs bhi (fee refund policy waghera)
PDF_PAGES = [
    (
        "https://islamabad.comsats.edu.pk/NEFiles/Notification-HEC-National-Fee-Refund-Policy-Revised-2024.pdf",
        "fees",
    ),
]


def decode_cf_email(cfemail: str) -> str:
    """Cloudflare email-obfuscation (XOR cipher) ko decode karta hai taake
    asli email address mile — warna sirf ek useless placeholder text
    ('[email protected]') extract hota hai jo koi real address nahi hai."""
    try:
        r = int(cfemail[:2], 16)
        return "".join(
            chr(int(cfemail[i : i + 2], 16) ^ r) for i in range(2, len(cfemail), 2)
        )
    except Exception:
        return ""


# Global footer/menu links jo LAGBHAG HAR page pe identical repeat hote hain
# (resource links, statutes list, downloads-modal items). Ye normal user ko
# bhi kam matlab ke hain aur RAG chunks mein sirf noise add karte hain.
FOOTER_JUNK_EXACT = {
    "resources",
    "hec refund policy",
    "pollens online",
    "sops for degree verification",
    "ug admission",
    "graduate admission",
    "student life",
    "rules & regulations",
    "hall reservation",
    "telephone directory",
    "readers society",
    "qec",
    "support",
    "office 365 setup",
    "secp guide",
    "statutory info",
    "cui act - 2018",
    "rules of business of the meetings of",
    "cui authorities - 2018",
    "statutes of the authorities, committees,",
    "principal officers, and teachers of cui - 2019",
    "elections",
    "follow us",
    "designed by",
    "it center",
    "close",
    "×",
}


def clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    for tag in soup.find_all(["nav", "header"]):
        tag.decompose()

    # Hidden/modal content (download-form popups, statutes text boxes) —
    # ye normal user ko screen pe kabhi nahi dikhte, isliye chatbot ke
    # knowledge base mein bhi nahi hone chahiye
    for tag in soup.find_all(attrs={"aria-hidden": "true"}):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.IGNORECASE)):
        tag.decompose()
    for selector in [
        {
            "class_": re.compile(
                r"(navbar|breadcrumb|sidebar|modal|popup|dialog)", re.IGNORECASE
            )
        },
        {
            "id": re.compile(
                r"(navbar|breadcrumb|sidebar|modal|popup|dialog)", re.IGNORECASE
            )
        },
    ]:
        for tag in soup.find_all(True, **selector):
            tag.decompose()

    # Cloudflare-obfuscated emails ko real email se replace karo (text
    # extract hone se PEHLE, warna placeholder hi reh jayega)
    for tag in soup.find_all(class_="__cf_email__"):
        cfemail = tag.get("data-cfemail")
        if cfemail:
            tag.replace_with(decode_cf_email(cfemail))
    for tag in soup.find_all("a", href=re.compile(r"/cdn-cgi/l/email-protection")):
        match = re.search(r"#([0-9a-fA-F]+)", tag.get("href", ""))
        if match:
            tag.replace_with(decode_cf_email(match.group(1)))

    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    boilerplate_patterns = [
        r"^(Home|News|Academics|Admissions|Research|Library|Health Care|Campus Life|About)$",
        r"^(Students|Faculty|Staff|Alumni|Apply|Visit)$",
        r"^©.*(All Rights Reserved|COMSATS University).*$",
    ]

    cleaned = []
    for l in lines:
        if any(re.match(p, l, re.IGNORECASE) for p in boilerplate_patterns):
            continue
        if l.lower() in FOOTER_JUNK_EXACT:
            continue
        cleaned.append(l)

    deduped = []
    for l in cleaned:
        if not deduped or deduped[-1] != l:
            deduped.append(l)
    return "\n".join(deduped)


def extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""


def url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def scrape():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    all_documents = []
    failed_urls = []

    for url, category in PAGES:
        resp = fetch_with_retry(session, url)
        if resp is None:
            failed_urls.append(url)
            time.sleep(REQUEST_DELAY)
            continue
        try:
            if resp.status_code != 200:
                log.warning(f"Skip (status {resp.status_code}): {url}")
                failed_urls.append(url)
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            title = extract_title(soup)
            content = clean_text(soup)

            if len(content) > 50:
                doc = {
                    "id": url_hash(url),
                    "url": url,
                    "type": "webpage",
                    "category": category,
                    "title": title,
                    "content": content,
                }
                all_documents.append(doc)
                log.info(f"[{category}] {url} ({len(content)} chars)")
            else:
                log.warning(
                    f"Skipped (too little content, {len(content)} chars): {url}"
                )
                failed_urls.append(url)
        except Exception as e:
            log.warning(f"Failed {url}: {e}")
            failed_urls.append(url)
        time.sleep(REQUEST_DELAY)

    for url, category in PDF_PAGES:
        resp = fetch_with_retry(session, url)
        if resp is None or resp.status_code != 200:
            failed_urls.append(url)
            time.sleep(REQUEST_DELAY)
            continue
        try:
            reader = PdfReader(io.BytesIO(resp.content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                doc = {
                    "id": url_hash(url),
                    "url": url,
                    "type": "pdf",
                    "category": category,
                    "title": os.path.basename(url),
                    "content": text.strip(),
                }
                all_documents.append(doc)
                log.info(f"[{category}][PDF] {url} ({len(text)} chars)")
            else:
                failed_urls.append(url)
        except Exception as e:
            log.warning(f"PDF failed {url}: {e}")
            failed_urls.append(url)
        time.sleep(REQUEST_DELAY)

    # Manually curated stable facts bhi add karo
    for fact in MANUAL_FACTS:
        all_documents.append(fact)
        log.info(f"[{fact['category']}][manual] {fact['title']}")

    with open(f"{OUTPUT_DIR}/all_documents.json", "w", encoding="utf-8") as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)

    log.info(
        f"DONE. {len(all_documents)} curated documents saved to {OUTPUT_DIR}/all_documents.json"
    )
    if failed_urls:
        log.warning(f"{len(failed_urls)} URL(s) failed / skipped:")
        for u in failed_urls:
            log.warning(f"  - {u}")
    else:
        log.info("All pages fetched successfully, no failures.")


if __name__ == "__main__":
    scrape()
