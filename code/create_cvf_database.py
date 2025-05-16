import requests
from bs4 import BeautifulSoup
import time
import sqlite3

def scrape_conference_papers(conference, years):
    base_url = "https://openaccess.thecvf.com"
    papers = []

    for year in years:
        # Correct URL pattern
        conference_url = f"{base_url}/{conference}{year}?day=all"
        print(f"Scraping: {conference_url}")
        try:
            response = requests.get(conference_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to retrieve {conference_url}: {e}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        paper_blocks = soup.find_all("dt")

        for dt in paper_blocks:
            title_tag = dt.find("a")
            if not title_tag or not title_tag.get("href", "").startswith("/content"):
                continue

            title = title_tag.text.strip()
            paper_url = base_url + title_tag["href"]

            # Pause between requests to avoid rate limiting
            time.sleep(0.01)

            try:
                paper_page = requests.get(paper_url, timeout=10)
                paper_page.raise_for_status()
            except requests.RequestException as e:
                print(f"Failed to load paper page {paper_url}: {e}")
                abstract = "Failed to load abstract"
            else:
                paper_soup = BeautifulSoup(paper_page.content, "html.parser")
                abstract_tag = paper_soup.find("div", id="abstract")
                abstract = abstract_tag.text.strip().replace("Abstract", "").strip() if abstract_tag else "No abstract found"

            papers.append({
                "title": title,
                "abstract": abstract,
                "url": paper_url,
                "conference": conference,
                "year": year
            })

    return papers


def scrape_all_conferences():
    conferences = ['CVPR', 'ICCV', 'ECCV']
    years = [2021, 2022, 2023, 2024]
    all_papers = []

    for conference in conferences:
        all_papers.extend(scrape_conference_papers(conference, years))

    return all_papers


# Example usage
if __name__ == "__main__":
    papers = scrape_all_conferences()

    for paper in papers[:3]:
        print(f"Title: {paper['title']}")
        print(f"Abstract: {paper['abstract']}")
        print(f"URL: {paper['url']}")
        print(f"Conference: {paper['conference']}, Year: {paper['year']}")
        print("="*100)


def initialize_db(db_name="cvf_papers.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            abstract TEXT,
            url TEXT UNIQUE,
            conference TEXT,
            year INTEGER
        )
    """)
    conn.commit()
    return conn

def insert_paper(conn, paper):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO papers (title, abstract, url, conference, year)
            VALUES (?, ?, ?, ?, ?)
        """, (paper["title"], paper["abstract"], paper["url"], paper["conference"], paper["year"]))
        conn.commit()
    except sqlite3.IntegrityError:
        # Paper already exists
        pass
     


if __name__ == "__main__":
    conn = initialize_db()
    conferences = ['CVPR', 'ICCV', 'ECCV']
    years = [2022, 2023, 2024]

    for conference in conferences:
        papers = scrape_conference_papers(conference, years)
        for paper in papers:
            insert_paper(conn, paper)

    # Example: Print the first 3 stored papers
    cursor = conn.cursor()
    for row in cursor.execute("SELECT title, abstract, url, conference, year FROM papers LIMIT 3"):
        print(f"Title: {row[0]}")
        print(f"Abstract: {row[1]}")
        print(f"URL: {row[2]}")
        print(f"Conference: {row[3]}, Year: {row[4]}")
        print("="*100)

    conn.close()
