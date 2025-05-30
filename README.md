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
    - `finding_similar_papers_directly_with_arxiv_module.ipynb` - experimental notebook
    - `finding_similar_papers_using_local_db.ipynb` - experimental notebook
    - `optirag.py` - baseline implementation script
    - `optirag_improved.py` - final implementation script

- **results/** 
    - `papers_to_check.txt` - a list of paper titles and queries we checked 
    - `preliminary_tests.docx` - opinion based test for the baseline model




## Current solution
- We begin by using the Python `arxiv` module to scrape paper titles, abstracts, and author information. 
- To continue the retrieval stage, we embed both the user query and the paper abstracts, and identify the most relevant papers using cosine similarity.
- We then generate a response by passing the context—composed of the five most similar paper summaries—to our base chatbot via the  `langchain` module
- For the base chatbot we are using the `Mistral-7B-Instruct-v0.2` quantized to 4 bits for efficiency. 
- For embedding extraction, we use the `allenai-specter` model, which is specifically designed for representing scientific documents.

## Current results
- To evaluate performance, we queried both the base chatbot and our RAG-enhanced model, then compared their responses.
- To highlight the improvement, we focused on papers or methods published after the release of the base chatbot. 
- The results are found in the `testing_th_model.docx` file. They are more description based and not so much definitive.

## Future directions
- Adding other data to score papers (like citations - use heuristics), that isn't available through the `arxiv` api
- Try to improve the prompt even further
- Currently the model can be used for any research questions, maybe we could somehow focus on a more specific field (for example by adding some keywords to the web search). 
- Maybe build a better embedding by combining more information than just the abstract
- Finding irrelevant results 
- Maybe we could have multiple models and switch based on the needs (accuracy vs speed)

