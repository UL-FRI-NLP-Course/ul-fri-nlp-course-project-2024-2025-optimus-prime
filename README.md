# Natural language processing course: `Conversational Agent  with Retrieval-Augmented Generation for research assistance`

## Contents
1. [Short description](#short-description)
2. [Team](#student-team)
3. [Reproducing the project](#reproducing-the-project)
4. [Structure of the repository](#structure-of-the-repository)

## Short description
This is a project implementing RAG to improve a chatbot's responses to queries about research papers, specifically targeting re-identification. For more details, check the report in `report.pdf` file.

## Student team
- Marko Medved
- Matej Vrečar
- Sebastijan Trojer

## Reproducing the project
- create a python environment (for example conda): `conda create --name myenv python=3.10`
- activate  the environment `conda activate myenv`
- install dependencies: `pip install -r requirements.txt`
- then you can run the scripts and notebooks in the **code** folder 
- The main script is `optirag_improved.py`, where you can use the chatbot improved with RAG interactively in the terminal
- If you want to use the local cvf database, you need to first run the web scraper: `create_cvf_database.py`

## Structure of the repository

- **code/**
    - `create_cvf_database.py` - create a local database by scraping cvf open access
    - `create_db_arxiv.py` - create a local database of papers scraped from arxiv (not used anymore since we can directly scrape when the user has a query)
    - `evaluation.ipynb` - code for calculating and plotting evaluation metrics
    - `finding_similar_papers_directly_with_arxiv_module.ipynb` - experimental notebook
    - `finding_similar_papers_using_local_db.ipynb` - experimental notebook
    - `optirag.py` - baseline implementation script
    - `optirag_improved.py` - final implementation script
    - `query_test.py` - short script to test a query
    - `utils.py` - utility functions for easier reuse

- **report/**
    - `report.pdf` - project report

- **results/** 
    - `papers_to_check.txt` - a list of paper titles and queries we checked 
    - `preliminary_tests.docx` - opinion based test for the baseline model
    - `queries` - all queries that were used to test paper retrieval
    - `results.xlsx` - results obtained from retrieval testing
