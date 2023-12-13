from typing import Dict, List, Type, Optional, Sequence
from typing_extensions import TypeAlias
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
from langchain.schema.output import LLMResult
from typing import Any, Dict, List, Optional, Mapping

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, root_validator, Extra
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pydantic import root_validator
from langchain.schema.messages import AnyMessage, BaseMessage, get_buffer_string
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessageChunk,
)

from langchain.schema import (
    ChatGeneration,
    ChatResult,
    LLMResult,
    PromptValue,
    RunInfo,
    Generation
)




class Llama(BaseLanguageModel):
    model_name: str
    llama_tokenizer: AutoTokenizer = Field(default=None)
    llama_model: AutoModelForCausalLM = Field(default=None)
    
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        print("initilization")
        print(self.model_name)
        self.setup_model_and_tokenizer()
        print(f'_tokenizer: {self.llama_tokenizer}')  # logging statement
        print(f'_model: {self.llama_model}')  # logging statement


    
    def setup_model_and_tokenizer(self):
        print("come there to build")
        print("tokenizer name: ", self.model_name)
        #self.llama_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, from_slow=True) #, add_eos_token=True, trust_remote_code=True)
        #print(self.llama_tokenizer)
        #self.llama_model = AutoModelForCausalLM.from_pretrained(
        #    self.model_name, device_map="auto", torch_dtype=torch.float16,
            #self.model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        #)
        local_model_path = '/share4/ru.wang/orca2_1epoch_small_lr/'
        self.llama_tokenizer = AutoTokenizer.from_pretrained(local_model_path, from_slow=True)
        self.llama_tokenizer.pad_token = '[PAD]'
        assert self.llama_tokenizer.pad_token_id == 32000
        
        # self.llama_model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name, trust_remote_code=True, torch_dtype=torch.float16
        # )
        print("load local weight")
        #'/work/gu14/k36037/code/llama-recipes/finetuned_7b'
        self.llama_model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.float16, device_map = "auto", use_flash_attention_2=True)
    
    # @root_validator(pre=True)
    # def setup_model_and_tokenizer(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    #     # Load tokenizer and model
    #     values['_tokenizer'] = AutoTokenizer.from_pretrained(values['model_name'])
    #     values['_model'] = AutoModelForCausalLM.from_pretrained(
    #         values['model_name'], trust_remote_code=True, torch_dtype=torch.float16
    #     )
    #     return values
    
    # {
    # "id": "chatcmpl-123",
    # "object": "chat.completion",
    # "created": 1677652288,
    # "model": "gpt-3.5-turbo-0613",
    # "choices": [{
    #     "index": 0,
    #     "message": {
    #     "role": "assistant",
    #     "content": "\n\nHello there, how may I assist you today?",
    #     },
    #     "finish_reason": "stop"
    # }],
    # "usage": {
    #     "prompt_tokens": 9,
    #     "completion_tokens": 12,
    #     "total_tokens": 21
    # }
    # }


    def generate_text(self, input_ids):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llama_model = self.llama_model.to(device)
        sliced_input_ids = input_ids[0][:4500].unsqueeze(0)
        input_ids = sliced_input_ids.to(device)
        #input_ids = input_ids.to(device)
        gen_params = {
            'max_length': min(len(input_ids) * 2.5, 4500),
            'temperature': 0.9,
            'repetition_penalty': 1.1,
            'top_p': 0.7,
            'top_k': 50
        }
        output = self.llama_model.generate(input_ids, **gen_params)
        output_text = self.llama_tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text


#llama-7b-32k 32000
#agent-lm-7b 10000
#agent-lm-13b 5000
    def generate_prompt(self, prompts: List[str], 
        stop: Optional[List[str]] = None,
        **kwargs: Any,) -> LLMResult:
        # This method assumes that each prompt in the list will receive a separate response.
        # It returns a dictionary where the keys are the prompts and the values are the generated responses.
        responses = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(prompts)
        #print(len(prompts))
        for prompt in prompts:
            #print(type(prompt), repr(prompt))
            #print(prompt)
            #print(len(prompt.text))
            #print(prompt)    
            input_ids = self.llama_tokenizer.encode(self.llama_tokenizer.bos_token + prompt.text, return_tensors="pt", max_length=4096, truncation=True)
            #input_ids = encoding['input_ids'] 
            sliced_input_ids = input_ids[0][:4096].unsqueeze(0)
            
            input_ids = sliced_input_ids.to(device)
            attention_mask = len(input_ids[0]) * [1]
            #attention_mask = attention_mask.to(device) 
            attention_mask = torch.tensor(attention_mask).to(device).unsqueeze(0)
            #input_ids[0] = input_ids[0][:32000]
            #input_ids = input_ids.to(device)
            #self.llama_model = self.llama_model.to(device)
            #len(input_ids[0])
            if "Based on the plan, please select the next action from the available action list" in prompt.text:
                print("make action, shrink the output length for valid output")
                max_l = int(2 * len(input_ids[0]))
            else:
                max_l = int(2 * len(input_ids[0]))
            print("len(input_ids[0]) = ", len(input_ids[0]))
            
            # encoding = self.llama_tokenizer.encode_plus(
            #         prompt.text, 
            #         return_tensors="pt", 
            #         max_length=32000, 
            #         truncation=True,
            #         padding="max_length"
            #     )
            
            output = self.llama_model.generate(
                input_ids,
                attention_mask = attention_mask,
                max_length=min(4096, max_l), #min(6000, 2 * len(input_ids[0])),  # or another value based on your needs
                temperature=0.7,
                repetition_penalty=1.1,
                no_repeat_ngram_size=5,
                top_p=0.7,
                top_k=50,
                do_sample=True,
            )
            

            #print(len(input_ids[0]), type(output))
            #print(repr(input_ids))
            output_text = self.llama_tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
            if "Based on the plan, please select the next action from the available action list" in prompt.text:
                print("make action original output is: ", output_text)
            #output_text = output_text.detach().cpu().numpy()
            print(f"input len {len(input_ids[0])}, output len {len(output[0])}")
            #{"token_usage": overall_token_usage, "model_name": self.model_name}for res in response["choices"]:
            # message = convert_dict_to_message(res["message"])
            # gen = ChatGeneration(
            #     message=message,
            #     generation_info=dict(finish_reason=res.get("finish_reason")),
            # )
            #ge = Generation(text = output_text, generation_info=None)
            #gen = [[ChatGeneration(message=BaseMessage(content = text, type = "model_returen"), generation_info=dict(finish_reason="finish_reason"))]]
            #gen = list(list({"text": output_text, "message": prompts}))
            gen = [[ChatGeneration(message=BaseMessage(content = output_text, type = "model_return"), generation_info=dict(finish_reason="stop"))]]
            llmoutput = {"token_usage": len(input_ids[0]) + len(output[0]), "model_name": self.model_name}
            responses = LLMResult(generations = gen, llm_output = llmoutput)
        return responses
    
    async def agenerate_prompt(self, prompts: List[str], *args, **kwargs) -> Dict[str, str]:
        # This method simply wraps the synchronous generate_prompt method for now,
        # as the provided code does not include asynchronous operations.
        return self.generate_prompt(prompts, *args, **kwargs)

    def invoke(self, input: str, *args, **kwargs) -> str:
        # This method takes a single input string and returns a single generated response string.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = self.llama_tokenizer.encode(input, return_tensors="pt", max_length=4096, truncation=True)
        input_ids = input_ids.to(device)
        sliced_input_ids = input_ids[0][:4096].unsqueeze(0)
        input_ids = sliced_input_ids.to(device)
        self.llama_tokenizer = self.llama_tokenizer.to(device)
        output = self.llama_tokenizer.generate(
            input_ids,
            max_length=min(4096, 2 * len(input_ids)),  # or another value based on your needs
            temperature=0.7,
            repetition_penalty=1.1,
            top_p=0.7,
            top_k=50
        )
        output_text = self.llama_tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text


    def predict(self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any) -> str:
        input_ids = self.llama_tokenizer.encode(text, return_tensors="pt")
        return self.generate_text(input_ids)

    def predict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any
    ) -> BaseMessage:
        text = " ".join([message.content for message in messages])
        response_text = self.predict(text, stop=stop, **kwargs)
        return AIMessage(content=response_text)

    async def apredict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        return self.predict(text, stop=stop, **kwargs)

    async def apredict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any
    ) -> BaseMessage:
        return self.predict_messages(messages, stop=stop, **kwargs)

    @property
    def InputType(self) -> TypeAlias:
        return str  # Assuming the input type is a string for this model.

    @property
    def _llm_type(self) -> str:
        return "Llama"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
        }

    # Implement other abstract methods from BaseLanguageModel as needed.




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
    "agentlm-7b": Llama,
    "agentlm-13b": Llama,
    "agentlm-13b": Llama,
    "zerooneai-6b": Llama,
    "zerooneai-34b": Llama,
    "Orca-2-13b": Llama,
    "Orca-2-7b": Llama,
    "zephyr-7b-beta": Llama,
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
    print("here")
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