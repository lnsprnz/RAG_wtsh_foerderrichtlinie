import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

urls = [
    'https://wtsh.de/de/foerderprogramme',
    'https://wtsh.de/de/foerderprogramm-aussenwirtschaftsfoerderung--gemeinschaftsbuero',
    'https://wtsh.de/de/einstiegsfoerderung-fuer-innovationsvorhaben-von-kmu-eik-transfer',
    'https://wtsh.de/de/bif-modul-1-prozess-und-organisationsinnovationen',
    'https://wtsh.de/de/bif-modul-2-entwicklungsvorhaben',
    'https://wtsh.de/de/bif-modul-3-komplexe-forschungs-und-entwicklungsvorhaben',
    'https://wtsh.de/de/einstiegsfoerderung-fuer-innovationsvorhaben-von-kmu-eik-seed',
    'https://wtsh.de/de/fit-verbundvorhaben',
    'https://wtsh.de/de/foerderung-einer-ressourceneffizienten-kreislaufwirtschaft',
    'https://wtsh.de/de/foerderung-energieeinspar-energieeffizienztechnologien-energieinnovationen-ehoch3',
    'https://wtsh.de/de/foerderung-niedrigschwelliger-innovativer-digitalisierungsmassnahmen-in-kleinen-unternehmen',
    'https://wtsh.de/de/foerderung-von-digitalen-spielen',
    'https://wtsh.de/de/ladeinfrastruktur-fuer-elektrofahrzeuge-2',
    'https://wtsh.de/de/energiewende-foerderaufruf',
    'https://wtsh.de/de/aufbau-einer-nachhaltigen-wasserstoffwirtschaft---wasserstoffrichtlinie'
]


def download_pdf(url, directory, downloaded_files):
    filename = os.path.basename(url)
    if not filename.lower().endswith('.pdf'):
        filename += '.pdf'

    filepath = os.path.join(directory, filename)

    # Check if the file has already been downloaded
    if filename in downloaded_files:
        print(f"Skipping {filename}, already downloaded.")
        return

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad requests

        # Write the response content to a file
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
        downloaded_files.add(filename)  # Add filename to the set of downloaded files
    except requests.RequestException as e:
        print(f"Failed to download {url}. Error: {e}")


def fetch_pdfs_from_url(url, directory, downloaded_files):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a')
        pdf_urls = [urljoin(url, link['href']) for link in links if
                    'href' in link.attrs and link['href'].endswith('.pdf')]

        for pdf_url in pdf_urls:
            download_pdf(pdf_url, directory, downloaded_files)
    except requests.RequestException as e:
        print(f"Failed to retrieve {url}. Error: {e}")


def scrape_and_download(urls, download_dir):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Initialize the set with files already in the directory to avoid re-downloads across script runs
    downloaded_files = set(os.listdir(download_dir))

    for url in urls:
        fetch_pdfs_from_url(url, download_dir, downloaded_files)


# Directory to store downloaded PDFs
download_directory = "downloaded_pdfs"

# Scrape and download PDFs
scrape_and_download(urls, download_directory)