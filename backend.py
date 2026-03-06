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
from groq import Groq
from dotenv import load_dotenv
import uvicorn

# =====================================================
# LOAD ENV VARIABLES
# =====================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-8b-instant"

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

def llm_call(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# =====================================================
# FIND OFFICIAL WEBSITE
# =====================================================

def find_official_domain(company):
    with DDGS() as ddgs:
        results = ddgs.text(f"{company} official website", max_results=5)
        for r in results:
            url = r.get("href", "")
            domain = urlparse(url).netloc.lower()
            if domain and "linkedin" not in domain:
                return "https://" + domain
    return ""

# =====================================================
# SCRAPE PAGE
# =====================================================

def scrape_page(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text[:4000]
    except:
        return ""

# =====================================================
# FIND INTERNAL LINKS
# =====================================================

def find_internal_links(base_url):
    pages = {"about": "", "team": "", "leadership": ""}
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(base_url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"].lower()
            full_url = urljoin(base_url, href)
            if "about" in href and pages["about"] == "":
                pages["about"] = full_url
            if "team" in href and pages["team"] == "":
                pages["team"] = full_url
            if "leadership" in href and pages["leadership"] == "":
                pages["leadership"] = full_url
    except:
        pass
    return pages

# =====================================================
# MANAGEMENT EXTRACTION
# =====================================================

def extract_management(text):
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    names = []
    for line in text.split("."):
        if "CEO" in line or "Chief" in line or "Director" in line:
            names.append(line.strip())
    return names[:10], emails[:10]

# =====================================================
# NEWS SEARCH
# =====================================================

def get_news(company):
    snippets = ""
    with DDGS() as ddgs:
        results = ddgs.text(
            f"{company} regulatory investigation breach cybersecurity",
            max_results=5
        )
        for r in results:
            snippets += r.get("title", "") + " " + r.get("body", "") + "\n"
    return snippets[:2000]

# =====================================================
# CAPABILITY EXTRACTION FROM PDF
# =====================================================

def get_capability_summary(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    system_prompt = """You are a senior advisory partner. Extract structured service capabilities from the document. Do not invent capabilities."""

    user_prompt = f"""Extract:
1 Core services
2 Industries served
3 Risk domains
4 Strategic differentiators

Document:
{text}"""

    return llm_call(system_prompt, user_prompt)

# =====================================================
# COMPANY ANALYSIS
# =====================================================

def analyze_company(company, website, about, team, news, capabilities):
    system_prompt = """You are a senior EY advisory partner. Use ONLY services from capability summary. Do not invent services."""

    user_prompt = f"""Target Company: {company}

Website:
{website}

About Section:
{about}

Leadership / Team:
{team}

News Signals:
{news}

EY Capabilities:
{capabilities}

Provide:
1 Company Summary
2 Observed Vulnerabilities
3 Relevant EY Services
4 Why these services are relevant now"""

    return llm_call(system_prompt, user_prompt)

# =====================================================
# PITCH EMAIL
# =====================================================

def generate_pitch(company, analysis, capabilities):
    system_prompt = """
You are a senior EY consulting partner with 50 years of experience in the consulting industry writing a strategic outreach email.

The services mentioned must ONLY come from the provided capability summary.

Write like a consulting advisory partner reaching out to a potential client.
The email should feel insightful and tailored to the company.
"""

    user_prompt = f"""
Target Company: {company}

Advisory Analysis:
{analysis}

EY Capability Summary:
{capabilities}

Write a personalized outreach email with the following structure:

Subject Line

Paragraph 1:
Brief overview of the company and the vulnerabilities / risk signals observed.
Explain what potential issues or operational exposures may exist.

Paragraph 2:
Explain why addressing these risks is important at this moment (regulatory pressure,
cyber risk landscape, operational complexity, etc.).

Section: How EY Can Help
Provide bullet points of relevant EY services.

For each bullet include:
• Service Name
  - Why it is relevant for the company
  - Business benefit

Closing paragraph requesting a meeting.

Length: 200-250 words.
"""


    return llm_call(system_prompt, user_prompt)

# =====================================================
# MAIN ANALYSIS ENDPOINT
# =====================================================

@app.post("/analyze")
async def analyze(
    company: str = Form(...),
    pdf: UploadFile = File(...)
):
    try:
        pdf_bytes = await pdf.read()

        # Step 1: Capabilities from PDF
        capabilities = get_capability_summary(pdf_bytes)

        # Step 2: Find domain
        domain = find_official_domain(company)

        # Step 3: Scrape website
        website_text = scrape_page(domain)

        # Step 4: Internal links
        links = find_internal_links(domain)
        about_text = scrape_page(links["about"]) if links["about"] else ""
        team_text = scrape_page(links["team"]) if links["team"] else ""

        # Step 5: News
        news = get_news(company)

        # Step 6: Analysis
        analysis = analyze_company(
            company, website_text, about_text, team_text, news, capabilities
        )

        # Step 7: Pitch
        pitch = generate_pitch(company, analysis, capabilities)

        # Step 8: Extract management
        names, emails = extract_management(about_text + team_text)

        return JSONResponse({
            "domain": domain,
            "leadership": names,
            "emails": emails,
            "news": news,
            "analysis": analysis,
            "pitch": pitch,
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
