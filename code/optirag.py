# ## Larger model

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
import torch
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer 
import arxiv 
import warnings
from langchain.chains import LLMChain

warnings.filterwarnings("ignore")

class OptimimusRAG:
    def __init__(self, llm_model, device):
        self.llm_model = llm_model
        self.device = device
        
    def initialize_model(self):
        
        # Load a chat-capable model
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.llm_model,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # 8-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, # loading in 4 bit
            bnb_4bit_quant_type="nf4", # quantization type
            bnb_4bit_use_double_quant=True, # nested quantization
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.llm_model,
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.llm_model,
            config=model_config,
            quantization_config=bnb_config, # we introduce the bnb config here.
            device_map="auto",
        )
        model.eval()
        
        text_generator = pipeline(
            task="text-generation",
            model= model,
            tokenizer=self.tokenizer,
            # device=self.device,
            return_full_text=True,
            max_new_tokens=8192,
            repetition_penalty=1.1,
        )   
        
        # Model used for chatting
        self.llm = HuggingFacePipeline(pipeline=text_generator) 
        
        self.embedding_model = SentenceTransformer('allenai-specter') # It can be used to map the titles & abstracts of scientific publications to a vector space such that similar papers are close.
        
        self.arxiv_client = arxiv.Client()
        self.chain = self._get_prompt_chain()


    def _encode_query(self, query):
        """
        Encode the query using the embedding model.
        """
        return self.embedding_model.encode(query)
    

    def _query_arxiv(self, query):
        """
        Query the arxiv API for papers related to the query.
        Returns a list of papers and their summaries (used for similarity search).
        """
        search = arxiv.Search(
            query=query,
            max_results=100,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        
        results = list(self.arxiv_client.results(search))
        
        papers, summaries = [], []
        for result in results:
            title = result.title
            authors = ', '.join([author.name for author in result.authors])
            summary = result.summary
            url = f"https://arxiv.org/abs/{result.entry_id.split('/')[-1]}"
            
            papers.append({
                "title": title,
                "authors": authors,
                "summary": summary,
                "url": url
            })
            summaries.append(summary)
        
        return papers, summaries


    def find_similar_papers(self, query, k=5):
        """Given a query, find the most similar papers
            Returns a list of k papers sorted by similarity score [dict].
        """
        
        # Get papers and summaries from arxiv
        papers, summaries = self._query_arxiv(query)
        
        # Embed original user query
        embedded_query = self._encode_query([query])
        
        # Embed the summaries of retrieved papers
        summary_embeddings = self._encode_query(summaries)
        
        # Compute cosine similarity between the query and the summaries
        similarities = cosine_similarity(embedded_query, summary_embeddings)[0]
        
        # Save similarities with papers
        for i, paper in enumerate(papers):
            paper['similarity'] = similarities[i]

        result = sorted(papers, key=lambda x: x['similarity'], reverse=True)[:k]

        for i, p in enumerate(result):
            print(f"{p['title']} rank={i}  (sim={p['similarity']:.3f})")
            
        return result
        
        
    def create_context(self, papers):
        context = "\n\n".join(
            f"Title: {pap['title']}\nSummary: {pap['summary']}" for pap in papers
        )
        return context
    
    def _get_prompt_chain(self):
        PROMPT_TEMPLATE = """
        You are a helpful AI QA assistant, for answering querries about research methods.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. This is important!
        You will also be receiving a history of the conversation, so you can use it to answer the question. When using the history, be sure to use the previous question, to consider the context of the currect question.
        You will be given a context of the most relevant papers, and you should use them to answer the question.

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
    
    gpt = OptimimusRAG(
        llm_model="mistralai/Mistral-7B-Instruct-v0.2",
        device="cuda"
    )
    
    gpt.initialize_model()
    chat = [{"query": "", "response": ""}]
    i = 0

    provide_response = False

    while True:
    
        query = input("Sup king\n\n" if i == 0 else "What else do you want to know\n\n") # "What is the best way to train a neural network?"
        
    
        papers = gpt.find_similar_papers(chat[-1]["query"] + query, k=30)
        context = gpt.create_context(papers)

        if provide_response:
            rag_response = gpt.generate_response(context, query)     
            normal_response = gpt.generate_response("", query)     
            chat.append({"query": query, "response": rag_response})

            print("Rag response:")
            print("="*100)
            print(rag_response)
            print("Normal response:")
            print("="*100)
            print(normal_response)
            i += 1
    
