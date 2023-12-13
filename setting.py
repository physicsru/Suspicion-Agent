from pathlib import Path
from typing import Any, Dict, List, Type

from pydantic.v1 import BaseModel, BaseSettings, Extra
import os



class LLMSettings(BaseModel):
    """
    LLM/ChatModel related settings
    """

    type: str = "chatopenai"

    class Config:
        extra = Extra.allow


class EmbeddingSettings(BaseModel):
    """
    Embedding related settings
    """

    type: str = "openaiembeddings"

    class Config:
        extra = Extra.allow


class ModelSettings(BaseModel):
    """
    Model related settings
    """

    type: str = ""
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()

    class Config:
        extra = Extra.allow


class Settings(BaseSettings):
    """
    Root settings
    """

    name: str = "default"
    model: ModelSettings = ModelSettings()

    class Config:
        env_prefix = "suspicionagent_"
        env_file_encoding = "utf-8"
        extra = Extra.allow

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                #json_config_settings_source,
                env_settings,
                file_secret_settings,
            )


# ---------------------------------------------------------------------------- #
#                             Preset configurations                            #
# ---------------------------------------------------------------------------- #
class OpenAIGPT4Settings(ModelSettings):
    # NOTE: GPT4 is in waitlist
    type = "openai-gpt-4-0613"
    llm = LLMSettings(type="chatopenai", model="gpt-4-0613", max_tokens=3000,temperature=0.1,  request_timeout=120)
    embedding = EmbeddingSettings(type="openaiembeddings")

class OpenAIGPT432kSettings(ModelSettings):
    # NOTE: GPT4 is in waitlist
    type = "openai-gpt-4-32k-0613"
    llm = LLMSettings(type="chatopenai", model="gpt-4-32k-0613", max_tokens=2500)
    embedding = EmbeddingSettings(type="openaiembeddings")


class OpenAIGPT3_5TurboSettings(ModelSettings):
    type = "openai-gpt-3.5-turbo"
    llm = LLMSettings(type="chatopenai", model="gpt-3.5-turbo-16k-0613", max_tokens=2500)
    embedding = EmbeddingSettings(type="openaiembeddings")


class OpenAIGPT3_5TextDavinci003Settings(ModelSettings):
    type = "openai-gpt-3.5-text-davinci-003"
    llm = LLMSettings(type="openai", model_name="text-davinci-003", max_tokens=2500)
    embedding = EmbeddingSettings(type="openaiembeddings")

class Llama2_7b_Settings(ModelSettings):
    type = "llama2-7b"
    llm = LLMSettings(type="llamav2", model_name="togethercomputer/Llama-2-7B-32K-Instruct", max_tokens=32000)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class agentlm_7b_Settings(ModelSettings):
    type = "agentlm-7b"
    llm = LLMSettings(type="agentlm-7b", model_name="THUDM/agentlm-7b", max_tokens=32000)
    embedding = EmbeddingSettings(type="openaiembeddings")

class agentlm_13b_Settings(ModelSettings):
    type = "agentlm-13b"
    llm = LLMSettings(type="agentlm-13b", model_name="THUDM/agentlm-13b", max_tokens=32000)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class chatglm3_6b_Settings(ModelSettings):
    type = "chatglm3-6b"
    llm = LLMSettings(type="chatglm3-6b", model_name="THUDM/chatglm3-6b-32k", max_tokens=32000)
    embedding = EmbeddingSettings(type="openaiembeddings")


class zerooneai_6b_Settings(ModelSettings):
    type = "zerooneai-6b"
    llm = LLMSettings(type="zerooneai-6b", model_name="01-ai/Yi-6B", max_tokens=32000)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class Orca_2_13b_Settings(ModelSettings):
    type = "Orca-2-13b"
    llm = LLMSettings(type="Orca-2-13b", model_name="microsoft/Orca-2-13b", max_tokens=4096)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class Orca_2_7b_Settings(ModelSettings):
    type = "Orca-2-7b"
    llm = LLMSettings(type="Orca-2-7b", model_name="microsoft/Orca-2-7b", max_tokens=4096)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class zephyr_7b_beta_settings(ModelSettings):
    type = "zephyr-7b-beta"
    llm = LLMSettings(type="zephyr-7b-beta", model_name="HuggingFaceH4/zephyr-7b-beta", max_tokens=4096)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
#THUDM/chatglm3-6b
# ------------------------- Model settings registry ------------------------ #
model_setting_type_to_cls_dict: Dict[str, Type[ModelSettings]] = {
    "openai-gpt-4-0613": OpenAIGPT4Settings,
    "openai-gpt-4-32k-0613": OpenAIGPT432kSettings,
    "openai-gpt-3.5-turbo": OpenAIGPT3_5TurboSettings,
    "openai-gpt-3.5-text-davinci-003": OpenAIGPT3_5TextDavinci003Settings,
    "llama2-7b": Llama2_7b_Settings,
    "agentlm-7b": agentlm_7b_Settings, 
    "agentlm-13b": agentlm_13b_Settings, 
    "chatglm3-6b": chatglm3_6b_Settings,
    "zerooneai-6b": zerooneai_6b_Settings,
    "Orca-2-7b": Orca_2_7b_Settings,
    "Orca-2-13b": Orca_2_13b_Settings,
    "zephyr-7b-beta": zephyr_7b_beta_settings, 
}


def load_model_setting(type: str) -> ModelSettings:
    if type not in model_setting_type_to_cls_dict:
        raise ValueError(f"Loading {type} setting not supported")

    cls = model_setting_type_to_cls_dict[type]
    return cls()


def get_all_model_settings() -> List[str]:
    """Get all supported Embeddings"""
    return list(model_setting_type_to_cls_dict.keys())

