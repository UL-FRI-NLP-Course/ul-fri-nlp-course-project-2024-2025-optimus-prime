# Natural language processing course: `Conversational Agent  with Retrieval-Augmented Generation for research assistance`

## Reproducing the project
- create a python environment (for example conda): `conda create --name myenv python=3.10`
- activate  the environment `conda activate myenv`
- install dependencies: `pip install -r requirements.txt`
- then you can run the scripts and notebooks in the **code** folder 
- the main implementation is in the `optirag.py` script

## Current solution
- We begin by using the Python `arxiv` module to scrape paper titles, abstracts, and author information. 
- To continue the retrieval stage, we embed both the user query and the paper abstracts, and identify the most relevant papers using cosine similarity.
- We then generate a response by passing the context—composed of the five most similar paper summaries—to our base chatbot via the  `langchain` module
- For the base chatbot we are using the `Mistral-7B-Instruct-v0.2` quantized to 4 bits for efficiency. 
- For embedding extraction, we use the `allenai-specter` model, which is specifically designed for representing scientific documents.

## Current results
- To evaluate performance, we queried both the base chatbot and our RAG-enhanced model, then compared their responses.
- To highlight the improvement, we focused on papers or methods published after the release of the base chatbot. 

# TODO : input results

## Future directions
- Adding other data to score papers (like citations - use heuristics), that isn't available through the `arxiv` api
- Try to improve the prompt even further
- Currently the model can be used for any research questions, maybe we could somehow focus on a more specific field (for example by adding some keywords to the web search). 
- Maybe build a better embedding by combining more information than just the abstract
- Finding irrelevant results 
- Maybe we could have multiple models and switch based on the needs (accuracy vs speed)

