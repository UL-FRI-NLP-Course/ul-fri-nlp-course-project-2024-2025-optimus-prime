import arxiv
import sqlite3

# Create a connection to an SQLite database
conn = sqlite3.connect('arxiv_papers.db')
cursor = conn.cursor()

# Create a table to store the paper information
cursor.execute('''
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY,
    title TEXT,
    authors TEXT,
    summary TEXT,
    published_date TEXT,
    arxiv_url TEXT
)
''')

# Define your query
query = "person re-identification"

# Use arxivpy to search for papers (adjust max_results as needed)
search = arxiv.Search(query=query, max_results=100)
results = search.results()

# Insert the papers into the database
for result in results:
    title = result.title
    # Extract author names as strings
    authors = ', '.join([author.name for author in result.authors])
    summary = result.summary
    published_date = result.published
    arxiv_url = f"https://arxiv.org/abs/{result.entry_id.split('/')[-1]}"
    
    cursor.execute('''
    INSERT INTO papers (title, authors, summary, published_date, arxiv_url)
    VALUES (?, ?, ?, ?, ?)
    ''', (title, authors, summary, published_date, arxiv_url))

# Commit changes and close the database connection
conn.commit()
conn.close()

print("Database has been populated with arXiv papers related to 'person re-identification'.")
