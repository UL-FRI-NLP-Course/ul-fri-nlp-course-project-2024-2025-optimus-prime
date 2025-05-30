import sqlite3

# Create a connection to the SQLite database
conn = sqlite3.connect("cvf_papers.db")
cursor = conn.cursor()

# Select all the papers from the 'papers' table
cursor.execute("SELECT * FROM papers")

# Fetch all rows from the result of the query
papers = cursor.fetchall()

# Print the results
for i, paper in enumerate(papers):
    print(f"ID: {paper[0]}")
    print(f"Title: {paper[1]}")
    print(f"Abstract: {paper[2]}")
    print(f"URL: {paper[3]}")
    print(f"Conference: {paper[4]}")
    print(f"Year: {paper[5]}")
    print("-" * 80)
    if i >= 10:
        break

# Close the database connection
conn.close()
