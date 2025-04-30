# ## TODO 
# - maybe some better model for embedding extraction
# - better prompt for the chatbot 
# - somehow test the implementation
# - Adding citations to score the papers
# - somehow separate the user query and searching for papers on arxiv

# ## Generate a response by incoroprating the retrieved papers with a chatbot

# ## Larger model

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
import torch
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


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
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.llm_model,
            config=model_config,
            quantization_config=bnb_config, # we introduce the bnb config here.
            device_map="auto",
        )
        self.model.eval()
        
        self.pipeline    
