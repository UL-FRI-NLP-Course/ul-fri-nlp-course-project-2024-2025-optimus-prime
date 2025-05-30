## IMPROVED VERSION OF THE OPTIRAG

import sqlite3
import warnings

import arxiv
import torch
from langchain.chains import LLMChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from utils import get_param

warnings.filterwarnings("ignore")

ARXIV_URL = get_param("ARXIV_URL")
LLM_MODEL = get_param("LLM_MODEL")


class OptimimusRAG:
    def __init__(self, llm_model, device):
        self.llm_model = llm_model
        self.device = device
        self.db_path = "cvf_papers.db"

    def initialize_model(self):
        # Load a chat-capable model
        SENTENCE_TRANSFORMER_MODEL = get_param("SENTENCE_TRANSFORMER_MODEL")

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.llm_model,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # 8-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # loading in 4 bit
            bnb_4bit_quant_type="nf4",  # quantization type
            bnb_4bit_use_double_quant=True,  # nested quantization
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.llm_model,
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.llm_model,
            config=model_config,
            quantization_config=bnb_config,  # we introduce the bnb config here.
            device_map="auto",
        )
        model.eval()

        text_generator = pipeline(
            task="text-generation",
            model=model,
            tokenizer=self.tokenizer,
            # device=self.device,
            return_full_text=True,
            max_new_tokens=8192,
            repetition_penalty=1.1,
        )

        # Model used for chatting
        self.llm = HuggingFacePipeline(pipeline=text_generator)

        self.embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

        self.arxiv_client = arxiv.Client()
        self.chain = self._get_prompt_chain()

    def _encode_query(self, query):
        """
        Encode the query using the embedding model.
        """
        return self.embedding_model.encode(query)

    def _enrich_query(self, query):
        # NOTE: Enrich query
        """
        Enriching the query with the domain keywords
        """
        domain_keywords = ["person re-identification", "video", "person re-id"]
        return query + " " + " ".join(domain_keywords)

    # NOTE: add rewriting of the query berfore searching and before matching
    def rewrite_query(self, query: str, mode: str = "search") -> str:
        if mode == "search":
            instruction = (
                "Rewrite the following query into a concise academic-style search query for arXiv. "
                "Use only key technical terms, no full sentences, no instructions, no labels. "
                "Return a single-line keyword-style query only."
            )
        elif mode == "matching":
            instruction = (
                "Rewrite the following query for use in semantic similarity matching with academic abstracts. "
                "Return only a single line with technical keywords or short phrases. "
                "Avoid explanations, formatting, or full sentences."
            )
        else:
            raise ValueError("mode must be 'search' or 'matching'")

        prompt = f"""{instruction}

    Original query:
    {query}

    Rewritten query:"""

        rewritten = self.llm(prompt).replace("+", "").strip()

        # Extract content after "Rewritten query:"
        if "Rewritten query:" in rewritten:
            return rewritten.split("Rewritten query:", 1)[-1].strip()

        # Otherwise, fallback to first meaningful line
        for line in rewritten.splitlines():
            line = line.strip()
            if line and not line.lower().startswith("rewritten query"):
                return line

        return rewritten.strip()

    def _query_arxiv(self, query):
        """
        Query the arxiv API for papers related to the query.
        Returns a list of papers and their summaries (used for similarity search).
        """
        # NOTE rewrite the query for searching
        query = self.rewrite_query(query, mode="search")

        print(f"Query rewritten for searching: {query}")

        # NOTE enrich the query
        query = self._enrich_query(query)

        print(query)

        search = arxiv.Search(
            # NOTE: added this to focus more on computer vision
            query=query,  # f"cat:cs.CV AND ({query})",
            max_results=100,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )

        results = list(self.arxiv_client.results(search))

        papers, summaries = [], []
        for result in results:
            title = result.title
            authors = ", ".join([author.name for author in result.authors])
            summary = result.summary
            url = f"{ARXIV_URL}{result.entry_id.split('/')[-1]}"

            papers.append(
                {"title": title, "authors": authors, "summary": summary, "url": url}
            )
            summaries.append(f"{title} {summary}")

        return papers, summaries

    # NOTE: addition query the local db from cvf
    def _query_local_db(self, query):
        """
        Retrieve all papers from the local SQLite database.
        Returns list of dictionaries similar to arXiv format.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT title, abstract, url, conference, year FROM papers")
        rows = cursor.fetchall()
        conn.close()

        papers = []
        for row in rows:
            papers.append(
                {
                    "title": row[0],
                    "summary": row[1],
                    "url": row[2],
                    "conference": row[3],
                    "year": row[4],
                }
            )
        return papers

    def find_similar_papers(
        self,
        query,
        k: int = 5,
        resort_across_sources: bool = False,
        use_local: bool = False,
    ):
        """
        Return the top-k most similar papers from arXiv and optionally local DB.

        Parameters:
            query (str): Query string.
            k (int): Number of top papers to return from each source.
            resort_across_sources (bool): If True, sort all results together.
            use_local (bool): If True, include local DB results.
        """

        # --- 1. Fetch arXiv papers ----------------------------------------------------
        arxiv_papers, arxiv_summaries = self._query_arxiv(query)

        # print(arxiv_papers)

        # NOTE rewrite the query for matching
        query = self.rewrite_query(query, mode="matching")
        print(f"Query rewritten for matching: {query}")

        # --- 2. Encode query once -----------------------------------------------------
        embedded_query = self._encode_query([query])

        # --- 3. Compute arXiv similarities --------------------------------------------
        arxiv_embeds = self._encode_query(arxiv_summaries)
        arxiv_sims = cosine_similarity(embedded_query, arxiv_embeds)[0]
        for p, sim in zip(arxiv_papers, arxiv_sims):
            p["similarity"] = sim
            p["source"] = "arxiv"

        # --- 4. Process local DB if enabled -------------------------------------------
        top_local = []
        if (
            use_local
        ):  # NOTE USE THIS TO SHOW THE ABILITY TO FIND OTHER METHODS NOT AVAILABLE ON ARXIV (AS AN IMPROVEMENT)
            local_papers = self._query_local_db(query)
            local_summaries = [f"{p['title']} {p['summary']}" for p in local_papers]
            local_embeds = self._encode_query(local_summaries)
            local_sims = cosine_similarity(embedded_query, local_embeds)[0]
            for p, sim in zip(local_papers, local_sims):
                p["similarity"] = sim
                p["source"] = "local"
            top_local = sorted(
                local_papers, key=lambda x: x["similarity"], reverse=True
            )[:k]

        # --- 5. Select top-k arXiv papers ---------------------------------------------
        top_arxiv = sorted(arxiv_papers, key=lambda x: x["similarity"], reverse=True)[
            :k
        ]

        # --- 6. Combine and optionally re-sort ----------------------------------------
        result = top_arxiv + top_local
        if resort_across_sources:
            result.sort(key=lambda x: x["similarity"], reverse=True)

        for i, p in enumerate(result):
            print(
                f"[{p['source']}] {p['title']}  rank = {i} (sim={p['similarity']:.3f})"
            )

        return result

    def create_context(self, papers):
        context = "\n\n".join(
            f"Title: {pap['title']}\nSummary: {pap['summary']}" for pap in papers
        )
        return context

    def _get_prompt_chain(self):
        # NOTE CHANGED THIS TO THE MORE SPECIFIC ONE
        PROMPT_TEMPLATE = """
       You are an expert AI assistant specialized in person re-identification (Re-ID
       ) from video data. Your task is to help researchers understand and apply 
       methods related to video-based person Re-ID, including techniques like feature 
       extraction, temporal modeling, tracking, and metric learning.

        If you do not know the answer to a question, clearly state that you do not know.
        Do not attempt to fabricate an answer.

        You will be given relevant context extracted from recent academic papers 
        (including titles and abstracts). Use this context to accurately answer 
        the user’s query. If a previous query is present, use it to maintain continuity.
        ```
        {context}
        ```

        ### Question:
        {query}

        ### Answer:
        """

        prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template=PROMPT_TEMPLATE.strip(),
        )

        chain = LLMChain(
            llm=self.llm,
            prompt=prompt_template,
        )

        return chain

    def generate_response(self, context, query):
        """
        Generate a response using the LLM chain.
        """
        response = self.chain.run(context=context, query=query)
        return response


if __name__ == "__main__":

    gpt = OptimimusRAG(llm_model=LLM_MODEL, device="cuda")

    gpt.initialize_model()
    chat = [{"query": "", "response": ""}]
    i = 0

    provide_response = False

    while True:

        query = input("Sup king\n\n" if i == 0 else "What else do you want to know\n\n")

        papers = gpt.find_similar_papers(chat[-1]["query"] + query, k=30)
        context = gpt.create_context(papers)

        if provide_response:
            rag_response = gpt.generate_response(context, query)
            normal_response = gpt.generate_response("", query)
            chat.append({"query": query, "response": rag_response})

            print("Rag response:")
            print("=" * 100)
            print(rag_response)
            print("Normal response:")
            print("=" * 100)
            print(normal_response)
            i += 1
