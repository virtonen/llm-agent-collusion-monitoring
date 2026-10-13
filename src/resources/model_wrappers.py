from abc import ABC, abstractmethod
from typing import List, TypedDict, Optional
from openai import OpenAI, APIError
from anthropic import Anthropic, APIConnectionError, RateLimitError, APIStatusError
import time
import logging
import os
from tqdm import tqdm
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content, GenerationConfig
import google.api_core.exceptions

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def is_ollama_model(model_name: str) -> bool:
    """Local Ollama tags (qwen2.5:14b, ollama/qwen2.5:14b). OpenRouter ids contain a slash."""
    name = model_name.lower()
    if name.startswith("ollama/"):
        return True
    if "/" in model_name or name.endswith(":free"):
        return False
    if name.startswith(("gpt", "claude", "gemini")):
        return False
    return True


def is_nvidia_nim_model(model_name: str) -> bool:
    """NVIDIA build.nvidia.com ids (org/name, no :free suffix) when a NIM key is set."""
    if not os.getenv("NVIDIA_API_KEY"):
        return False
    if model_name.endswith(":free") or is_ollama_model(model_name):
        return False
    if model_name.lower().startswith(("gpt", "claude", "gemini")):
        return False
    return "/" in model_name

class Message(TypedDict):
    role: str
    content: str

class ModelWrapper(ABC):
    """Abstract base class for model API wrappers."""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        top_p: float = 1.0,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            max_retries: Maximum number of retry attempts
            initial_retry_delay: Initial delay between retries (seconds)
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.additional_params = kwargs

    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        """Generate a response for a single prompt."""
        pass

    @abstractmethod
    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
        """Generate responses for multiple prompts."""
        pass

    @classmethod
    def create(cls, model_name: str, **kwargs) -> 'ModelWrapper':
        """Factory method to create appropriate model wrapper instance."""
        if is_ollama_model(model_name):
            return OllamaModelWrapper(model_name, **kwargs)
        elif is_nvidia_nim_model(model_name):
            return NvidiaClient(model_name, **kwargs)
        elif "gpt" in model_name.lower():
            return OpenAIClient(model_name, **kwargs)
        elif "claude" in model_name.lower():
            return AnthropicClient(model_name, **kwargs)
        elif "gemini" in model_name.lower():
            return GoogleClient(model_name, **kwargs)
        else:
            return OpenRouterClient(model_name, **kwargs)

    def _exponential_backoff(self, attempt: int) -> None:
        """Implement exponential backoff between retries."""
        if attempt < self.max_retries:
            delay = self.initial_retry_delay * (2 ** attempt)
            time.sleep(delay)

class OpenAIClient(ModelWrapper):
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = OpenAI(timeout=60)
    
    def generate(self, messages: List[Message]) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    top_p=self.top_p,
                    **self.additional_params
                )
                return response.choices[0].message.content
            except APIError as e:
                logger.warning(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {str(e)} for prompt {messages}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    return None
                self._exponential_backoff(attempt)

    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
         #TODO: implement batch API for message_lists that are sufficiently long
        responses = []
        for messages in tqdm(messages_list, desc='Batch Generation'):
            responses.append(self.generate(messages))
        return responses

class OpenRouterClient(ModelWrapper):

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set. Add it to .env or the environment.")
        self.client = OpenAI(
            base_url = "https://openrouter.ai/api/v1",
            api_key = api_key,
            timeout=120,
        )
    
    def generate(self, messages: List[Message]) -> str:
        for attempt in range(self.max_retries):
            try:
                caller_params = {k: v for k, v in self.additional_params.items() if k != "extra_body"}
                extra_body = {"reasoning": {"enabled": False}}
                extra_body.update(self.additional_params.get("extra_body") or {})
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    top_p=self.top_p,
                    extra_body=extra_body,
                    **caller_params
                )
                return response.choices[0].message.content
            except APIError as e:
                logger.warning(f"OpenRouter API error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if getattr(e, "status_code", None) == 403 or attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    return None
                self._exponential_backoff(attempt)

    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
         #TODO: implement batch API for message_lists that are sufficiently long
        responses = []
        for messages in tqdm(messages_list, desc='Batch Generation'):
            responses.append(self.generate(messages))
        return responses


class NvidiaClient(ModelWrapper):
    """OpenAI-compatible client for NVIDIA NIM free endpoints (build.nvidia.com)."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY is not set. Create one at https://build.nvidia.com and add it to .env.")
        self.client = OpenAI(
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key=api_key,
            timeout=180,
        )

    def generate(self, messages: List[Message]) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = response.choices[0].message.content
                if content:
                    return content
                logger.warning(f"NVIDIA NIM returned empty content (attempt {attempt + 1}/{self.max_retries})")
            except APIError as e:
                logger.warning(f"NVIDIA NIM API error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if getattr(e, "status_code", None) in (401, 403, 404) or attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    return None
            if attempt == self.max_retries - 1:
                return None
            self._exponential_backoff(attempt)

    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
        responses = []
        for messages in tqdm(messages_list, desc='NVIDIA Batch Generation'):
            responses.append(self.generate(messages))
        return responses


class OllamaModelWrapper(ModelWrapper):
    """OpenAI-compatible client for a local Ollama server."""

    def __init__(self, model_name: str, **kwargs):
        if model_name.lower().startswith("ollama/"):
            model_name = model_name.split("/", 1)[1]
        super().__init__(model_name, **kwargs)
        self.client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
            timeout=180,
        )

    def generate(self, messages: List[Message]) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    **self.additional_params,
                )
                return response.choices[0].message.content
            except APIError as e:
                logger.warning(f"Ollama API error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    return None
                self._exponential_backoff(attempt)

    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
        responses = []
        for messages in tqdm(messages_list, desc='Ollama Batch Generation'):
            responses.append(self.generate(messages))
        return responses


class AnthropicClient(ModelWrapper):
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = Anthropic(timeout=60)
    
    def generate(self, messages: List[Message]) -> str:

        if messages[0]['role'] == 'system':
            system = messages[0]['content']
            messages = messages[1:]
        else:
            system = 'You are a helpful assistant.'

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    system=system,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    **self.additional_params
                )
                return response.content[0].text

            except (APIConnectionError, RateLimitError, APIStatusError) as e:
                logger.warning(f"Anthropic API error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)} for prompt {messages}")
                    return None
                self._exponential_backoff(attempt)

    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
        #TODO: implement batch API for message_lists that are sufficiently long
        responses = []
        for messages in tqdm(messages_list, desc='Batch Generation'):
            responses.append(self.generate(messages))
        return responses


class GoogleClient(ModelWrapper):

    def __init__(
        self,
        model_name: str,
        project: Optional[str] = os.getenv("GOOGLE_PROJECT_ID"),
        location: str = "us-central1",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        if project is None:
            raise ValueError("Google Project ID must be provided via argument or GOOGLE_PROJECT_ID env variable.")
        vertexai.init(project=project, location=location)
        print(f"project is {project}")
        self.model = GenerativeModel(model_name=self.model_name)

    def _convert_messages_to_contents(self, messages: List[Message]) -> List[Content]:
        """Converts OpenAI/Anthropic style messages to Vertex AI Content objects."""
        contents = []
        for message in messages:
            role = message['role']
            if role == 'assistant':
                role = 'model'
            if role in ['user', 'model']:
                contents.append(Content(role=role, parts=[Part.from_text(message['content'])]))
        return contents

    def generate(self, messages: List[Message]) -> str:
        system_prompt = None
        if messages and messages[0]['role'] == 'system':
            system_prompt = messages[0]['content']
            messages = messages[1:]

        contents = self._convert_messages_to_contents(messages)

        generation_config = GenerationConfig(
            temperature=self.temperature,
            # max_output_tokens=self.max_tokens,
            max_output_tokens=65535,
            top_p=self.top_p,
        )

        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    contents=contents,
                    generation_config=generation_config,
                )
                if response.candidates and response.candidates[0].content.parts:
                     return response.candidates[0].content.parts[0].text
                else:
                    logger.warning(f"Google API returned no content (attempt {attempt + 1}/{self.max_retries}) for prompt {messages}. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'N/A'}")
                    return None

            except (google.api_core.exceptions.RetryError,
                    google.api_core.exceptions.ServiceUnavailable,
                    google.api_core.exceptions.InternalServerError,
                    google.api_core.exceptions.ResourceExhausted,
                    google.api_core.exceptions.InvalidArgument) as e:
                logger.warning(f"Google API error (attempt {attempt + 1}/{self.max_retries}): {str(e)} for prompt {messages}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    return None
                self._exponential_backoff(attempt)
            except Exception as e:
                logger.error(f"Unexpected error during Google API call: {str(e)} for prompt {messages}")
                return None

    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
        responses = []
        for messages in tqdm(messages_list, desc='Google Batch Generation'):
            responses.append(self.generate(messages))
        return responses
