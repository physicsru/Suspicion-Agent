from typing import Dict, List, Type, Optional

from langchain import chat_models, embeddings, llms
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLanguageModel

from setting import EmbeddingSettings, LLMSettings
from context import Context
from setting import Settings
from rich.console import Console
from agent import SuspicionAgent
from pydantic import BaseModel, Extra, Field, root_validator
#from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from typing import Any, Dict, List, Optional, Mapping

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, root_validator, Extra
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Llama(BaseModel):
    model_name: str
    _tokenizer: AutoTokenizer = None
    _model: AutoModelForCausalLM = None

    class Config:
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def setup_model_and_tokenizer(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Load tokenizer and model
        values['_tokenizer'] = AutoTokenizer.from_pretrained(values['model_name'])
        values['_model'] = AutoModelForCausalLM.from_pretrained(
            values['model_name'], trust_remote_code=True, torch_dtype=torch.float16
        )
        return values

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)
        input_ids = input_ids.to(device)

        gen_params = {
            'max_length': min(len(input_ids) * 2, 32000),
            'temperature': 0.7,
            'repetition_penalty': 1.1,
            'top_p': 0.7,
            'top_k': 50
        }

        output = self._model.generate(input_ids, **gen_params)
        output_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            'tokenizer_model_name': self.tokenizer_model_name,
            'model_name': self.model_name,
            'trust_remote_code': self.trust_remote_code,
            'torch_dtype': self.torch_dtype
        }



def agi_init(
    agent_configs: List[dict],
    game_config:dict,
    console: Console,
    settings: Settings,
    user_idx: int = 0,
    webcontext=None,
) -> Context:
    ctx = Context(console, settings, webcontext)
    ctx.print("Creating all agents one by one...", style="yellow")
    for idx, agent_config in enumerate(agent_configs):
        agent_name = agent_config["name"]
        with ctx.console.status(f"[yellow]Creating agent {agent_name}..."):
            agent = SuspicionAgent(
                name=agent_config["name"],
                age=agent_config["age"],
                rule=game_config["game_rule"],
                game_name=game_config["name"],
                observation_rule=game_config["observation_rule"],
                status="N/A",  
                llm=load_llm_from_config(ctx.settings.model.llm),
            
                reflection_threshold=8,
            )
            for memory in agent_config["memories"]:
                agent.add_memory(memory)
        ctx.robot_agents.append(agent)
        ctx.agents.append(agent)
    
        ctx.print(f"Agent {agent_name} successfully created", style="green")

    ctx.print("Suspicion Agent started...")
   
    return ctx


# ------------------------- LLM/Chat models registry ------------------------- #
llm_type_to_cls_dict: Dict[str, Type[BaseLanguageModel]] = {
    "chatopenai": chat_models.ChatOpenAI,
    "openai": llms.OpenAI,
    "llamav2": Llama,
}

# ------------------------- Embedding models registry ------------------------ #
embedding_type_to_cls_dict: Dict[str, Type[Embeddings]] = {
    "openaiembeddings": embeddings.OpenAIEmbeddings
}


# ---------------------------------------------------------------------------- #
#                                LLM/Chat models                               #
# ---------------------------------------------------------------------------- #
def load_llm_from_config(config: LLMSettings) -> BaseLanguageModel:
    """Load LLM from Config."""
    config_dict = config.dict()
    config_type = config_dict.pop("type")

    if config_type not in llm_type_to_cls_dict:
        raise ValueError(f"Loading {config_type} LLM not supported")
    print("load here")
    cls = llm_type_to_cls_dict[config_type]
    print("load there")
    return cls(**config_dict)


def get_all_llms() -> List[str]:
    """Get all supported LLMs"""
    return list(llm_type_to_cls_dict.keys())


# ---------------------------------------------------------------------------- #
#                               Embeddings models                              #
# ---------------------------------------------------------------------------- #
def load_embedding_from_config(config: EmbeddingSettings) -> Embeddings:
    """Load Embedding from Config."""
    config_dict = config.dict()
    config_type = config_dict.pop("type")
    print(config)
    if config_type not in embedding_type_to_cls_dict:
        raise ValueError(f"Loading {config_type} Embedding not supported")

    cls = embedding_type_to_cls_dict[config_type]
    return cls(**config_dict)


def get_all_embeddings() -> List[str]:
    """Get all supported Embeddings"""
    return list(embedding_type_to_cls_dict.keys())

