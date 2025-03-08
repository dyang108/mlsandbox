import requests
import pandas as pd
import os
import time
from tqdm import tqdm
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Constants
SEC_SEARCH_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={}&type={}&dateb=&owner=exclude&count=100"
HEADERS = {"User-Agent": "Your Name (your.email@example.com)"}
SAVE_DIR = "sec_filings_pdfs"
FILING_TYPES = ["10-K", "10-Q"]
START_YEAR = 2014
END_YEAR = 2024
MAX_WORKERS = 1  # Number of threads

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Step 1: Load S&P 500 Companies (with CIK)
def load_sp500_companies():
    """Loads a list of S&P 500 companies, tickers, and CIKs"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_df = tables[0]
    
    # Ensure CIK is a string, drop non-numeric values, and pad with leading zeros
    sp500_df["CIK"] = sp500_df["CIK"].astype(str).str.extract(r"(\d+)")  # Extract only numbers
    sp500_df = sp500_df.dropna(subset=["CIK"])  # Drop rows where CIK is still NaN
    sp500_df["CIK"] = sp500_df["CIK"].apply(lambda x: x.zfill(10))  # Pad CIK to 10 digits

    return sp500_df[["Symbol", "Security", "CIK"]].rename(columns={"Symbol": "Ticker", "Security": "Company"})

# Step 2: Fetch Filing URLs from SEC Search
def get_filing_links(cik, filing_type):
    """Scrapes the SEC Edgar search page for filing links"""
    print(cik, filing_type)
    url = SEC_SEARCH_URL.format(str(cik).zfill(10), filing_type)
    print(url)
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    filing_links = []

    for row in soup.find_all("tr"):
        print(row)
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        
        filing_date = cells[3].text.strip()
        if not filing_date:
            continue

        year = int(filing_date[:4])
        if not (START_YEAR <= year <= END_YEAR):
            continue

        detail_link = row.find("a", {"id": "documentsbutton"})
        if detail_link:
            filing_links.append(("https://www.sec.gov" + detail_link["href"], filing_date))

    return filing_links

# Step 3: Extract PDF Link from Filing Page
def get_pdf_link(filing_page_url):
    """Extracts the PDF link from the filing details page"""
    response = requests.get(filing_page_url, headers=HEADERS)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        
        doc_link = cells[2].find("a")
        if doc_link and "pdf" in doc_link.text.lower():
            return "https://www.sec.gov" + doc_link["href"]

    return None

# Step 4: Download and Save PDF
def download_pdf(pdf_url, company, filing_type, filing_date):
    """Downloads the PDF file and saves it"""
    response = requests.get(pdf_url, headers=HEADERS)
    if response.status_code == 200:
        filename = f"{company}_{filing_type}_{filing_date}.pdf".replace(" ", "_")
        filepath = os.path.join(SAVE_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filename
    return None

# Step 5: Worker Function for Multithreading
def process_company_filing(company, cik):
    """Processes a single company's filings using multithreading"""
    results = []
    print(cik, company)
    for filing_type in FILING_TYPES:
        filing_links = get_filing_links(cik, filing_type)

        for filing_page_url, filing_date in filing_links:
            pdf_url = get_pdf_link(filing_page_url)
            print(filing_type, filing_date, pdf_url)
            if pdf_url:
                filename = download_pdf(pdf_url, company, filing_type, filing_date)
                if filename:
                    results.append({"Company": company, "Filing": filing_type, "Date": filing_date, "File": filename})
            time.sleep(1)  # Rate-limiting per request

    return results

# Step 6: Run Scraping with Multithreading
def scrape_sp500_filings_pdfs():
    """Main function to scrape 10-K and 10-Q PDFs using multithreading"""
    sp500_companies = load_sp500_companies()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_company = {
            executor.submit(process_company_filing, row["Company"], row["CIK"]): row["Company"]
            for _, row in sp500_companies.iterrows()
        }

        for future in tqdm(as_completed(future_to_company), total=len(sp500_companies)):
            try:
                results.extend(future.result())
            except Exception as e:
                print(f"Error processing {future_to_company[future]}: {e}")
                # traceback.print_exc()

    # Save metadata
    results_df = pd.DataFrame(results)
    results_df.to_csv("sp500_filings_pdfs_metadata.csv", index=False)
    print("Scraping completed. Data saved.")

# Run the scraper
scrape_sp500_filings_pdfs()
