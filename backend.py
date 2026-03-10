from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from urllib.parse import urlparse, urljoin
import fitz
import re
import os
import datetime
import wikipedia
from groq import Groq
from dotenv import load_dotenv
import uvicorn

# =====================================================
# LOAD ENV VARIABLES
# =====================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Startup guard ──────────────────────────────────
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. "
        "Create a .env file in the project root with: GROQ_API_KEY=gsk_..."
    )
if not GROQ_API_KEY.startswith("gsk_"):
    raise RuntimeError(
        f"GROQ_API_KEY looks malformed (got: {GROQ_API_KEY[:8]}...). "
        "It should start with 'gsk_'."
    )
# ───────────────────────────────────────────────────

client = Groq(api_key=GROQ_API_KEY)
MODEL  = "llama-3.1-8b-instant"

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(title="AI Intelligence Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LLM CALL (GROQ)
# =====================================================

def llm_call(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# =====================================================
# FIND OFFICIAL WEBSITE
# =====================================================

def find_official_domain(company: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(f"{company} official website", max_results=5)
        for r in results:
            url    = r.get("href", "")
            domain = urlparse(url).netloc.lower()
            if domain and "linkedin" not in domain:
                return "https://" + domain
    return ""

# =====================================================
# SCRAPE PAGE
# =====================================================

def scrape_page(url: str) -> str:
    try:
        headers  = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=10, headers=headers)
        soup     = BeautifulSoup(response.text, "html.parser")
        text     = soup.get_text(separator=" ", strip=True)
        return text[:4000]
    except:
        return ""

# =====================================================
# FIND INTERNAL LINKS
# =====================================================

def find_internal_links(base_url: str) -> dict:
    pages = {"about": "", "team": "", "leadership": ""}
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r       = requests.get(base_url, headers=headers, timeout=10)
        soup    = BeautifulSoup(r.text, "html.parser")
        for link in soup.find_all("a", href=True):
            href     = link["href"].lower()
            full_url = urljoin(base_url, href)
            if "about"      in href and pages["about"]      == "":
                pages["about"]      = full_url
            if "team"       in href and pages["team"]       == "":
                pages["team"]       = full_url
            if "leadership" in href and pages["leadership"] == "":
                pages["leadership"] = full_url
    except:
        pass
    return pages

# =====================================================
# CXO TITLES REFERENCE LIST
# =====================================================

CXO_TITLES = [
    "CEO", "CTO", "CFO", "COO", "CISO", "CMO", "CRO",
    "Chief Executive", "Chief Technology", "Chief Financial",
    "Chief Operating", "Chief Information", "Chief Marketing",
    "Chief Risk", "Chief Security", "Chief Data",
    "Chief Analytics", "Chief AI", "Chief Digital",
    "President", "Managing Director", "Executive Director",
    "Vice President", "Head of Data", "Head of Analytics",
    "Head of AI", "Head of Machine Learning", "Director of Data",
    "Director of Analytics", "Director of AI", "Head of"
]

# =====================================================
# EXTRACT EMAILS WITH CONTEXT WINDOW
# =====================================================

def extract_emails_with_context(text: str, source_label: str) -> list:
    """
    Finds all emails in text and tries to associate each one with
    a nearby CXO name/title using a sliding sentence context window.
    Returns list of dicts: {name, title, email, source}
    """
    results   = []
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)

    for i, sentence in enumerate(sentences):
        emails_found = re.findall(
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", sentence
        )
        if not emails_found:
            continue

        context_window = " ".join(sentences[max(0, i - 2): i + 3])

        for email in emails_found:
            if any(skip in email.lower() for skip in [
                "noreply", "no-reply", "support", "info", "contact",
                "hello", "admin", "webmaster", "privacy", "legal", "press"
            ]):
                continue

            name_found  = ""
            title_found = ""

            for title in CXO_TITLES:
                if title.lower() in context_window.lower():
                    title_found  = title
                    name_pattern = re.findall(
                        rf"([A-Z][a-z]+ [A-Z][a-z]+)(?=.*{re.escape(title)})|"
                        rf"(?<={re.escape(title)}\s)([A-Z][a-z]+ [A-Z][a-z]+)",
                        context_window
                    )
                    for match in name_pattern:
                        candidate = match[0] or match[1]
                        if candidate:
                            name_found = candidate
                            break
                    break

            results.append({
                "name":   name_found,
                "title":  title_found,
                "email":  email,
                "source": source_label
            })

    return results

# =====================================================
# CXO EMAILS — FROM COMPANY WEBSITE
# =====================================================

def get_cxo_emails_from_website(base_url: str, company: str) -> list:
    all_emails     = []
    candidate_urls = [base_url]
    keywords       = [
        "about", "team", "leadership", "management",
        "executives", "contact", "people"
    ]

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r       = requests.get(base_url, headers=headers, timeout=10)
        soup    = BeautifulSoup(r.text, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"].lower()
            if any(kw in href for kw in keywords):
                full_url = urljoin(base_url, link["href"])
                if full_url not in candidate_urls:
                    candidate_urls.append(full_url)
    except:
        pass

    for url in candidate_urls[:8]:
        try:
            resp      = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            page_text = BeautifulSoup(resp.text, "html.parser").get_text(separator=" ", strip=True)
            found     = extract_emails_with_context(page_text, f"Company Website ({url})")
            all_emails.extend(found)
        except:
            continue

    seen   = set()
    unique = []
    for entry in all_emails:
        if entry["email"] not in seen:
            seen.add(entry["email"])
            unique.append(entry)

    return unique

# =====================================================
# CXO EMAILS — FROM WIKIPEDIA
# =====================================================

def get_cxo_emails_from_wikipedia(company: str) -> list:
    results   = []
    cxo_names = []

    try:
        wikipedia.set_lang("en")
        search_results = wikipedia.search(company, results=3)
        if not search_results:
            return results

        page = None
        for title in search_results:
            try:
                candidate = wikipedia.page(title, auto_suggest=False)
                if company.lower().split()[0] in candidate.title.lower():
                    page = candidate
                    break
            except:
                continue

        if not page:
            return results

        content = page.content

        direct_emails = extract_emails_with_context(content, f"Wikipedia ({page.url})")
        results.extend(direct_emails)

        for title in CXO_TITLES:
            patterns = [
                rf"{re.escape(title)}\s+([A-Z][a-z]+ [A-Z][a-z]+)",
                rf"([A-Z][a-z]+ [A-Z][a-z]+)\s*[,\(]?\s*{re.escape(title)}",
            ]
            for pat in patterns:
                matches = re.findall(pat, content)
                for match in matches:
                    name = match.strip()
                    if name and len(name.split()) >= 2:
                        cxo_names.append((name, title))

        seen_names = set()
        unique_cxo = []
        for name, title in cxo_names:
            if name not in seen_names:
                seen_names.add(name)
                unique_cxo.append((name, title))

        results.append({
            "_meta":                    True,
            "cxo_names_from_wikipedia": unique_cxo[:15],
            "wikipedia_url":            page.url
        })

    except Exception:
        pass

    return results

# =====================================================
# INFER EMAIL PATTERNS FROM CXO NAMES + DOMAIN
# =====================================================

def infer_cxo_emails(cxo_names: list, domain: str) -> list:
    inferred     = []
    clean_domain = (
        domain
        .replace("https://", "")
        .replace("http://",  "")
        .lstrip("www.")
    )

    for name, title in cxo_names:
        parts = name.strip().split()
        if len(parts) < 2:
            continue
        first = parts[0].lower()
        last  = parts[-1].lower()

        patterns = [
            f"{first}.{last}@{clean_domain}",
            f"{first[0]}{last}@{clean_domain}",
            f"{first}@{clean_domain}",
            f"{last}@{clean_domain}",
            f"{first}{last}@{clean_domain}",
        ]

        for email_pattern in patterns:
            inferred.append({
                "name":     name,
                "title":    title,
                "email":    email_pattern,
                "source":   f"Inferred pattern — Wikipedia leadership data (domain: {clean_domain})",
                "inferred": True
            })

    return inferred

# =====================================================
# FULL CXO INTELLIGENCE GATHERING
# =====================================================

def gather_cxo_intelligence(company: str, domain: str) -> dict:
    confirmed_emails = []
    inferred_emails  = []
    wiki_cxo_names   = []

    if domain:
        confirmed_emails = get_cxo_emails_from_website(domain, company)

    wiki_results = get_cxo_emails_from_wikipedia(company)
    for entry in wiki_results:
        if entry.get("_meta"):
            wiki_cxo_names = entry.get("cxo_names_from_wikipedia", [])
        else:
            confirmed_emails.append(entry)

    if wiki_cxo_names and domain:
        inferred_emails = infer_cxo_emails(wiki_cxo_names, domain)

    return {
        "confirmed": confirmed_emails,
        "inferred":  inferred_emails,
        "cxo_names": [{"name": n, "title": t} for n, t in wiki_cxo_names]
    }

# =====================================================
# LEGACY MANAGEMENT EXTRACTION (backward compatibility)
# =====================================================

def extract_management(text: str):
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    names  = []
    for line in text.split("."):
        if any(t in line for t in ["CEO", "Chief", "Director", "President"]):
            names.append(line.strip())
    return names[:10], emails[:10]

# =====================================================
# NEWS SEARCH — DATA / ML / ANALYTICS FOCUSED
# =====================================================

def get_news(company: str) -> str:
    """
    Pulls news signals specifically around data strategy, analytics maturity,
    AI/ML adoption, data governance failures, and digital transformation —
    not cybersecurity.
    """
    snippets = ""
    queries = [
        f"{company} data strategy analytics artificial intelligence 2024 2025",
        f"{company} machine learning AI transformation digital 2024 2025",
        f"{company} data governance data quality data management 2024 2025",
        f"{company} predictive analytics business intelligence reporting 2024 2025",
        f"{company} data lake data warehouse cloud migration 2024 2025",
    ]
    with DDGS() as ddgs:
        for q in queries:
            try:
                results = ddgs.text(q, max_results=2)
                for r in results:
                    snippets += r.get("title", "") + " " + r.get("body", "") + "\n"
            except:
                continue
    return snippets[:3000]

# =====================================================
# LITIGATION INTELLIGENCE — DATA / PRIVACY / AI FOCUSED
# =====================================================

def get_litigation_info(company: str) -> list:
    """
    Searches for regulatory actions, fines, and legal disputes specifically
    around data privacy, AI/ML misuse, data governance failures, algorithmic
    bias, and data protection violations — last 6 months.
    """
    search_queries = [
        f"{company} data privacy violation fine penalty 2024 2025",
        f"{company} GDPR DPDP data protection regulatory action",
        f"{company} AI algorithm bias discrimination lawsuit",
        f"{company} data breach privacy class action settlement",
        f"{company} data governance failure audit regulatory 2024 2025",
        f"{company} algorithmic decision making investigation penalty",
    ]

    DATA_LITIGATION_KEYWORDS = [
        "data privacy", "data breach", "gdpr", "dpdp", "data protection",
        "algorithm", "ai bias", "machine learning", "data governance",
        "privacy violation", "data misuse", "consent violation",
        "algorithmic", "data fine", "data penalty", "data lawsuit",
        "lawsuit", "penalty", "fine", "enforcement", "violation",
        "settlement", "investigation", "regulatory", "court", "judgment",
        "compliance", "charged", "probe", "action", "audit"
    ]

    all_snippets = []

    with DDGS() as ddgs:
        for query in search_queries:
            try:
                hits = ddgs.text(query, max_results=3)
                for r in hits:
                    title    = r.get("title", "")
                    body     = r.get("body",  "")
                    url      = r.get("href",  "")
                    combined = (title + " " + body).lower()
                    if any(kw in combined for kw in DATA_LITIGATION_KEYWORDS):
                        all_snippets.append({
                            "title":      title,
                            "summary":    body[:400],
                            "url":        url,
                            "query_used": query
                        })
            except:
                continue

    seen_titles = set()
    unique      = []
    for s in all_snippets:
        if s["title"] not in seen_titles:
            seen_titles.add(s["title"])
            unique.append(s)

    return unique[:10]

# =====================================================
# CAPABILITY EXTRACTION FROM PDF
# =====================================================

def get_capability_summary(pdf_bytes: bytes) -> str:
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    system_prompt = """You are a senior advisory partner specialising in Data & Analytics,
Data Science, Machine Learning, and Data Governance consulting.
Extract structured service capabilities from the document.
Do not invent capabilities. Be precise, structured, and comprehensive.
Pay special attention to any Data, Analytics, AI/ML, Statistics, or Governance services."""

    user_prompt = f"""Extract the following from the document:
1. Core Services (focus on Data, Analytics, AI/ML, Data Science, Data Governance services)
2. Industries Served
3. Data & Analytics Risk / Maturity Domains Covered
4. Strategic Differentiators in Data & AI
5. Notable Methodologies, Frameworks, or Tools (e.g. DAMA, DCAM, MLOps, DataOps, CRISP-DM)

Document:
{text}"""

    return llm_call(system_prompt, user_prompt)

# =====================================================
# COMPANY ANALYSIS — DATA / ML / GOVERNANCE FOCUSED
# =====================================================

def analyze_company(
    company:      str,
    website:      str,
    about:        str,
    team:         str,
    news:         str,
    capabilities: str,
    litigation:   list
) -> str:
    """
    Produces a strategic advisory assessment focused entirely on the company's
    data maturity, analytics capability gaps, ML/AI readiness, data governance
    posture, and statistical decision-making quality.
    Cybersecurity is NOT the primary lens — data value & governance is.
    """

    if litigation:
        litigation_text = ""
        for item in litigation:
            litigation_text += (
                f"• {item['title']}\n"
                f"  {item['summary']}\n"
                f"  Source: {item['url']}\n\n"
            )
    else:
        litigation_text = "No significant data/privacy litigation signals found in the last 6 months."

    system_prompt = """You are a senior EY advisory partner and subject-matter expert in:
- Data Strategy & Analytics
- Data Science & Statistical Modelling
- Machine Learning & AI implementation
- Data Governance, Data Quality & Master Data Management
- Advanced Analytics, Business Intelligence & Reporting
- MLOps, DataOps & AI/ML productionisation
- Regulatory compliance around data (GDPR, DPDP, CCPA, AI Act)

Your analysis must be EXCLUSIVELY focused on the company's data, analytics, AI/ML,
and data governance landscape. Do NOT default to cybersecurity.
Use ONLY services from the provided capability summary. Do not invent services or facts."""

    user_prompt = f"""Target Company: {company}

Website Content:
{website}

About Section:
{about}

Leadership / Team (note any CDO, Chief Data Officer, Head of Analytics, Head of AI, etc.):
{team}

Data & Analytics News Signals:
{news}

Data Privacy / Governance Litigation & Regulatory Actions (Last 6 Months):
{litigation_text}

EY Capabilities:
{capabilities}

Provide a detailed DATA & ANALYTICS strategic advisory assessment. Structure it as follows:

1. COMPANY DATA PROFILE
   - What data assets does this company likely sit on? (transactional, customer, operational, financial)
   - Current observable state of data maturity based on web presence, tech stack signals, job postings
   - Whether they have a CDO / Chief Data function visible
   - Industry-specific data complexity (e.g. regulatory reporting, real-time decisioning needs)

2. DATA STRATEGY GAPS & SIGNALS
   - Evidence of fragmented or siloed data infrastructure
   - Absence or presence of a unified data strategy
   - Signals of underutilised data assets (manual reporting, spreadsheet dependency, legacy BI)
   - Any observable lag vs. industry peers in analytics maturity

3. AI / MACHINE LEARNING READINESS
   - Current AI/ML signals from news, website, job postings, or products
   - Gaps between their stated AI ambitions and likely ground reality
   - MLOps and model governance maturity signals
   - Risks from ungoverned or unvalidated AI/ML models in production

4. DATA GOVERNANCE & QUALITY EXPOSURE
   - Data governance posture (GDPR, DPDP Act, CCPA compliance signals)
   - Data quality issues that could be inferred from sector and scale
   - Master Data Management gaps
   - Data lineage and metadata management maturity
   - Regulatory reporting accuracy risks

5. STATISTICAL & ANALYTICAL DECISION-MAKING QUALITY
   - Are business decisions being made on solid statistical foundations?
   - Risks from poor A/B testing, sampling bias, or anecdotal decision-making
   - Reporting accuracy and KPI consistency risks

6. MOST RELEVANT EY DATA & ANALYTICS SERVICES
   - Map ONLY from the provided EY capability summary
   - Each service must be tied directly to a specific gap identified above

7. WHY THESE SERVICES ARE URGENT NOW
   - Regulatory timelines (AI Act, DPDP, GDPR enforcement trends)
   - Competitive pressure from data-native peers
   - Board-level scrutiny on AI/ML ROI and data investment returns

8. QUANTIFIED RISK OF INACTION
   - Revenue at risk from poor data quality
   - Regulatory fine exposure from data governance failures
   - Strategic disadvantage from delayed AI/ML adoption"""

    return llm_call(system_prompt, user_prompt)

# =====================================================
# HIGH-CONVERSION PITCH EMAIL — DATA / ML / ANALYTICS FOCUSED
# =====================================================

def generate_pitch(
    company:      str,
    analysis:     str,
    capabilities: str,
    cxo_name:     str  = "",
    litigation:   list = None
) -> str:
    """
    Generates a high-conversion outreach email laser-focused on:
    - Data Strategy & Analytics
    - Data Science & Machine Learning
    - Data Governance & Data Quality
    - Advanced Statistics & Decision Intelligence
    - AI/ML productionisation and governance

    NOT cybersecurity. NOT generic risk.
    """

    salutation = f"Dear {cxo_name}," if cxo_name else "Dear [Executive Name],"

    litigation_context = ""
    if litigation:
        litigation_context = (
            "\n\nRecent Data & Regulatory Signals "
            "(reference risk category — not the case directly):\n"
        )
        for item in litigation[:5]:
            litigation_context += f"• {item['title']} — {item['summary'][:200]}\n"

    system_prompt = """You are a senior EY advisory partner and world-class B2B communicator
with 30 years of experience in Data & Analytics, Data Science, Machine Learning,
Data Governance, and Advanced Statistics consulting.

You have converted hundreds of Fortune 500 and mid-market companies into long-term
EY data and analytics advisory clients.

YOUR EMAIL MUST BE:
- EXCLUSIVELY about Data Strategy, Data Science, Machine Learning, Data Governance,
  Advanced Analytics, Statistical Modelling, and AI/ML adoption
- DO NOT write about cybersecurity, cyber risk, or information security
- DO NOT write generic risk or compliance content that could apply to any company
- WRITE with the depth and specificity of someone who has studied this company's
  data landscape before picking up the pen

YOUR WRITING PHILOSOPHY:
- Lead with a SPECIFIC DATA INSIGHT — something that reveals you understand their
  data maturity, analytics challenges, or AI/ML situation
- Show the GAP between where they are and where data-mature competitors already are
- Position EY as the partner who has already solved this exact problem for comparable companies
- Make the VALUE of better data / ML / analytics TANGIBLE in business outcome terms
  (faster decisions, revenue from data products, reduced cost from data quality, etc.)
- Create urgency through REGULATORY DATA REQUIREMENTS (AI Act, DPDP, GDPR, model risk),
  COMPETITIVE INTELLIGENCE, and BOARD-LEVEL AI ROI expectations
- End with a no-friction offer that a data leader would say yes to immediately

HARD RULES:
- Services mentioned MUST come ONLY from the provided capability summary
- Do NOT invent facts, statistics, or incidents not present in the analysis
- NEVER mention cybersecurity, cyber risk, or information security as primary themes
- NEVER use "in today's rapidly evolving landscape" or similar filler
- Every paragraph must feel written specifically for THIS company's data situation"""

    user_prompt = f"""Target Company: {company}

Salutation: {salutation}

Data & Analytics Advisory Analysis (research-backed — use this as the factual backbone):
{analysis}
{litigation_context}

EY Capability Summary (ONLY use services listed here — do not invent):
{capabilities}

Write a high-conversion outreach email.
The email MUST be focused on Data Strategy, Data Science, Machine Learning,
Data Governance, Advanced Analytics, and Statistical Decision-Making.
Do NOT include section headers in the final email — it must read as one flowing professional letter.
Do NOT limit the length. Be comprehensive, specific, and authoritative.

STRUCTURE TO FOLLOW:

[SUBJECT LINE]
Must reference something specific about the company's DATA or AI/ML situation.
Reference an observable signal — a data product, an analytics gap, a regulatory deadline,
or a competitive data advantage their peers already have.
Example styles:
  "Re: {company}'s Data Governance Readiness — A Strategic Perspective"
  "Re: Closing the Analytics Gap — An EY Perspective for {company}"
  "Re: AI/ML Productionisation at {company} — Where the Value Is Being Left Behind"

[OPENING — The Data Insight Hook]
Open with a sharp, specific observation about THIS company's data landscape.
What does their website, product, job postings, or news tell you about their
data maturity stage? Are they still in spreadsheet-driven decision-making?
Do they have a CDO? Are they investing in AI/ML without the governance foundation?
Do NOT open with pleasantries. Do NOT mention cybersecurity.
Make the reader think: "How do they know that about our data situation?"

[THE DATA MATURITY NARRATIVE — The Gap That's Costing Them]
Tell the story of where they are vs. where they need to be.
Cover at least THREE of the following data dimensions specific to their situation:
  - Data quality and consistency across systems
  - Fragmented analytics and siloed reporting
  - AI/ML models running without governance or validation frameworks
  - Statistical decision-making quality and bias in analytics
  - Data governance posture under GDPR, DPDP, or AI Act
  - Underutilised data assets that peers are already monetising
  - The gap between "we have data" and "we generate value from data"
Make the business cost of these gaps tangible — in revenue, cost, speed, or regulatory risk terms.

[THE COMPETITIVE CONTEXT — Why Peers Are Already Ahead]
One focused paragraph on what data-mature competitors in their sector are already doing
that this company is not. Reference sector-specific data use cases where relevant
(e.g. predictive maintenance in manufacturing, next-best-action in financial services,
demand forecasting in retail, patient outcome modelling in healthcare, etc.).
Make the reader feel the urgency of the gap without manufactured alarm.

[THE CREDIBILITY BRIDGE — Why EY]
Briefly establish EY's specific depth in Data & Analytics advisory.
Reference EY's data science capabilities, governance frameworks (DAMA, DCAM, etc.),
AI/ML productionisation track record, and sector-specific data expertise.
Be matter-of-fact and confident — not boastful.

[HOW EY CAN HELP — Data & Analytics Value Proposition]
Present 4–6 EY services, each directly mapped to a specific data/ML/governance gap.
Use ONLY services from the provided capability summary.
For each service:
  • [Service Name]
    - The specific data problem it solves for {company}
    - The measurable business outcome (faster decisions, reduced data debt,
      model accuracy improvement, regulatory readiness, revenue from data)
    - Why this is the right moment to act (regulatory deadline, competitive window,
      board AI mandate, data debt compound interest)

[THE COST OF DATA INACTION]
One powerful paragraph on what happens if these data gaps are not addressed.
Be factual: reference data quality cost benchmarks (e.g. Gartner's $12.9M/year
average cost of poor data quality), regulatory fine trajectories under GDPR/AI Act,
and the compounding cost of technical data debt.
Do NOT reference cybersecurity incidents.

[CALL TO ACTION — High-Value, Low-Friction]
Offer one of:
  - A complimentary Data Maturity Assessment (30–45 minutes)
  - A focused AI/ML Governance Readiness Briefing
  - A Data Strategy Workshop scoped to their specific sector

Suggest 2–3 time windows. Make it easy to say yes.
Frame it as insight delivery, not a sales call.

[PROFESSIONAL CLOSING]
Warm, authoritative close from a senior EY Data & Analytics advisory partner.
Name placeholder, title (e.g. Partner, Data & Analytics Advisory),
practice area, and contact details placeholder."""

    return llm_call(system_prompt, user_prompt)

# =====================================================
# MAIN ANALYSIS ENDPOINT
# =====================================================

@app.post("/analyze")
async def analyze(
    company: str        = Form(...),
    pdf:     UploadFile = File(...)
):
    try:
        pdf_bytes = await pdf.read()

        # Step 1 — Capability summary from uploaded PDF
        capabilities = get_capability_summary(pdf_bytes)

        # Step 2 — Discover official domain
        domain = find_official_domain(company)

        # Step 3 — Scrape main website
        website_text = scrape_page(domain)

        # Step 4 — Scrape internal subpages
        links      = find_internal_links(domain)
        about_text = scrape_page(links["about"]) if links["about"] else ""
        team_text  = scrape_page(links["team"])  if links["team"]  else ""

        # Step 5 — Data & Analytics news intelligence
        news = get_news(company)

        # Step 6 — Data / Privacy litigation intelligence (last 6 months)
        litigation = get_litigation_info(company)

        # Step 7 — CXO email intelligence (website + Wikipedia + inferred)
        cxo_intel = gather_cxo_intelligence(company, domain)

        # Step 8 — Full data & analytics strategic analysis
        analysis = analyze_company(
            company, website_text, about_text, team_text,
            news, capabilities, litigation
        )

        # Step 9 — Identify top CXO for personalised salutation
        # Prefer data-focused leaders: CDO > CTO > CEO
        top_cxo_name = ""
        data_titles  = [
            "Chief Data", "Chief Analytics", "Chief AI", "Chief Digital",
            "Head of Data", "Head of Analytics", "Head of AI",
            "CTO", "Chief Technology", "CEO", "Chief Executive"
        ]

        if cxo_intel["cxo_names"]:
            for preferred_title in data_titles:
                for entry in cxo_intel["cxo_names"]:
                    if preferred_title.lower() in entry["title"].lower():
                        top_cxo_name = entry["name"]
                        break
                if top_cxo_name:
                    break
            if not top_cxo_name:
                top_cxo_name = cxo_intel["cxo_names"][0]["name"]

        # Step 10 — Generate high-conversion data & analytics pitch email
        pitch = generate_pitch(
            company, analysis, capabilities,
            cxo_name=top_cxo_name,
            litigation=litigation
        )

        # Step 11 — Legacy management extraction (fallback)
        names, emails = extract_management(about_text + team_text)

        # ── Build & return response ──────────────────────────
        return JSONResponse({
            "domain": domain,

            # Legacy fields (backward compatibility)
            "leadership": names,
            "emails":     emails,

            # CXO intelligence
            "cxo_intelligence": {
                "confirmed_emails": cxo_intel["confirmed"],
                "inferred_emails":  cxo_intel["inferred"],
                "cxo_names":        cxo_intel["cxo_names"],
                "note": (
                    "confirmed_emails are directly scraped from the company website or Wikipedia. "
                    "inferred_emails are algorithmically generated using common email patterns "
                    "based on names from Wikipedia — they are NOT verified. "
                    "Always verify before use in any client communication."
                )
            },

            # News
            "news": news,

            # Litigation & data governance regulatory signals
            "litigation": {
                "count":   len(litigation),
                "signals": litigation,
                "note": (
                    "Results filtered to last 6 months — focused on data privacy, "
                    "AI/ML regulation, data governance failures, and data protection enforcement. "
                    "Each item should be independently verified before use in client communications."
                )
            },

            # Core outputs
            "analysis":     analysis,
            "pitch":        pitch,
            "capabilities": capabilities
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok"}

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
