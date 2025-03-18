import os
import csv
import time
import requests
from urllib.parse import urlparse

# Constants
CSV_FILE = '../data/company-filings/pdfs.txt'
OUTPUT_DIR = '../data/company-filings/filings'
HEADERS = {"User-Agent": "Your Name (derickwyang@gmail.com)"}
REQUEST_DELAY = 0.11  # Delay between requests (to avoid rate limiting)
REQUEST_TIMEOUT = 1  # Timeout in seconds for each request

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Open and read the CSV file
with open(CSV_FILE, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        file_id = row['id']
        url = row['web_link']
        if int(file_id) < 28200:
            continue

        # Parse URL to determine file extension, default to ".pdf" if none found
        parsed_url = urlparse(url)
        _, ext = os.path.splitext(parsed_url.path)
        if not ext:
            ext = ".pdf"
        
        # Construct output file path (e.g., "filings/5.pdf")
        output_path = os.path.join(OUTPUT_DIR, f"{file_id}{ext}")

        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {url} -> {output_path}")
            else:
                print(f"Failed to download {url}: HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"Skipped {url} (Timed out after {REQUEST_TIMEOUT} sec)")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

        # Pause between requests to respect potential rate limits
        time.sleep(REQUEST_DELAY)
