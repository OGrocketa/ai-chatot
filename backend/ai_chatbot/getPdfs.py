import requests
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import json

load_dotenv()
pdf_endpoint_api = os.environ.get("PDF_ENDPOINT")

def scrape_lesson_links(page_url):
    """
    Scrapes the provided page URL for lesson links.
    Adjust the filtering criteria (e.g., href pattern) as needed.
    """
    response = requests.get(page_url)
    if response.status_code != 200:
        print(f"Error fetching page {page_url}: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    lesson_links = []

    # Loop through all <a> tags that have an href attribute.
    # Customize the filtering criteria to match the lesson links.
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Example filtering: select only relative links or links that contain a keyword
        if href.startswith("/") or "lesson" in href.lower():
            full_url = requests.compat.urljoin(page_url, href)
            lesson_links.append(full_url)
    
    # Remove duplicates
    return list(set(lesson_links))


def get_save_path(page_url):
    """
    Creates a file name for the PDF based on the URL path.
    Instead of creating subdirectories, it joins the path components with an underscore.
    For example:
      - "https://www.promptingguide.ai/introduction" becomes "introduction.pdf"
      - "https://www.promptingguide.ai/introduction/basics" becomes "introduction_basics.pdf"
    All PDFs are saved in the same directory (e.g. "pdfs").
    """
    parsed = urlparse(page_url)
    # Remove leading/trailing slashes from the path
    path = parsed.path.strip("/")
    
    if not path:
        filename = "index.pdf"
    else:
        parts = path.split("/")
        filename = "_".join(parts) + ".pdf"
    
    # Save all PDFs in the "pdfs" directory
    return os.path.join("pdfs", filename)


def download_pdf_from_url(page_url):
    """
    Sends the provided URL to the PDF conversion API,
    then downloads and saves the PDF locally.
    """
    convert_api_url = "https://api.pdfendpoint.com/v1/convert"
    payload = {
        "url": page_url  # The URL of the webpage to convert
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {pdf_endpoint_api}"
    }
    
    # Request the PDF conversion
    response = requests.post(convert_api_url, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        # Extract the PDF URL from the JSON result
        pdf_url = result["data"]["url"]
        # (The API returns a filename, but we override it with our URL-based name.)
        
        # Retrieve the actual PDF file
        pdf_response = requests.get(pdf_url)
        if pdf_response.status_code == 200:
            # Determine the file path based on the original page URL.
            file_path = get_save_path(page_url)
            # Ensure the directory exists.
            os.makedirs("pdfs", exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(pdf_response.content)
            print(f"PDF saved successfully for {page_url} at {file_path}")
        else:
            print(f"Error retrieving PDF file for {page_url}: {pdf_response.status_code} - {pdf_response.text}")
    else:
        print(f"Error converting {page_url}: {response.status_code} - {response.text}")


def main():
    # Define the URL of the page that contains the lesson links
    base_page_url = "https://www.promptingguide.ai/introduction/settings"
    
    # Scrape the lesson links from the base page
    lesson_links = scrape_lesson_links(base_page_url)
    print(f"Found {len(lesson_links)} lesson links.")

    with open("lesson_links.json", "w") as f:
        json.dump(lesson_links, f, indent=2)
    print("Lesson links saved to lesson_links.json.")

    # Loop over each lesson link and download its PDF version
    for i in range(0,90):
        download_pdf_from_url(lesson_links[i])


if __name__ == "__main__":
    main()
