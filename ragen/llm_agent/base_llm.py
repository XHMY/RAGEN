from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any, Tuple
import os
import asyncio
import time

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from together import AsyncTogether

@dataclass
class LLMResponse:
    """Unified response format across all LLM providers"""
    content: str
    model_name: str

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response from the LLM"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation"""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        if "o1-mini" in self.model_name:
            if messages[0]["role"] == "system":
                messages = messages[1:]
            
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        if response.choices[0].finish_reason in ['length', 'content_filter']:
            raise ValueError("Content filtered or length exceeded")
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model
        )

class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider implementation"""
    
    def __init__(self, model_name: str = "deepseek-reasoner", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided and not found in environment variables")
        
        self.client = AsyncOpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        if "o1-mini" in self.model_name:
            if messages[0]["role"] == "system":
                messages = messages[1:]
            
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        if response.choices[0].finish_reason in ['length', 'content_filter']:
            raise ValueError("Content filtered or length exceeded")
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model
        )

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider implementation
    Refer to https://github.com/anthropics/anthropic-sdk-python
    """
    
    def __init__(self, model_name: str = "claude-3.5-sonnet-20240620", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and not found in environment variables")
        
        self.client = AsyncAnthropic(api_key=self.api_key)
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        # Extract system message if present
        system_content = ""
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                # Map to Anthropic's format
                chat_messages.append({
                    "role": "assistant" if msg["role"] == "assistant" else "user",
                    "content": msg["content"]
                })
        
        response = await self.client.messages.create(
            model=self.model_name,
            system=system_content,
            messages=chat_messages,
            **kwargs
        )
        if response.stop_reason == "max_tokens":
            raise ValueError("Max tokens exceeded")
        return LLMResponse(
            content=response.content[0].text,
            model_name=response.model
        )

class TogetherProvider(LLMProvider):
    """Together AI API provider implementation"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3-70b-chat-hf", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together API key not provided and not found in environment variables")
        
        self.client = AsyncTogether(api_key=self.api_key)
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model
        )

class VllmLocalProvider(LLMProvider):
    """VLLM Local API provider implementation (for local VLLM servers)"""
    
    def __init__(self, model_name: str = "ragen_model", base_url: str = "http://localhost:8000/v1"):
        self.model_name = model_name
        self.base_url = base_url
        self.client = AsyncOpenAI(
            api_key="token-abc123",  # VLLM doesn't require real API key
            base_url=base_url
        )
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model
        )

class LocalVLLMProvider(LLMProvider):
    """Local VLLM API provider implementation for localhost servers"""
    
    def __init__(self, base_url: str, model_name: str = "ragen_model", api_key: str = "dummy_key"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.api_key = api_key
        
        # Create OpenAI client pointing to local VLLM server
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key  # VLLM servers don't validate this
        )
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            
            if response.choices[0].finish_reason in ['length', 'content_filter']:
                raise ValueError("Content filtered or length exceeded")
                
            return LLMResponse(
                content=response.choices[0].message.content,
                model_name=response.model
            )
        except Exception as e:
            # Add context about the local server
            raise ValueError(f"Local VLLM server error (URL: {self.base_url}): {e}")

class ConcurrentLLM:
    """Unified concurrent interface for multiple LLM providers"""
    
    def __init__(self, provider: Union[str, LLMProvider], model_name: Optional[str] = None, 
                api_key: Optional[str] = None, max_concurrency: int = 4, base_url: Optional[str] = None):
        """
        Initialize the concurrent LLM client.
        
        Args:
            provider: Either a provider instance or a string ('openai', 'anthropic', 'together', 'vllm_local')
            model_name: Model name (if provider is a string)
            api_key: API key (if provider is a string)
            max_concurrency: Maximum number of concurrent requests
            base_url: Base URL for local VLLM servers (only used with 'vllm_local' provider)
        """
        if isinstance(provider, LLMProvider):
            self.provider = provider
        else:
            if provider.lower() == "openai":
                self.provider = OpenAIProvider(model_name or "gpt-4o", api_key)
            elif provider.lower() == "deepseek":
                self.provider = DeepSeekProvider(model_name or "deepseek-reasoner", api_key)
            elif provider.lower() == "anthropic":
                self.provider = AnthropicProvider(model_name or "claude-3-7-sonnet-20250219", api_key)
            elif provider.lower() == "together":
                self.provider = TogetherProvider(model_name or "meta-llama/Llama-3-70b-chat-hf", api_key)
            elif provider.lower() == "vllm_local":
                if not base_url:
                    raise ValueError("base_url is required for vllm_local provider")
                self.provider = LocalVLLMProvider(base_url, model_name or "ragen_model", api_key or "dummy_key")
            else:
                raise ValueError(f"Unknown provider: {provider}")
        
        # Store max_concurrency but don't create the semaphore yet
        self.max_concurrency = max_concurrency
        self._semaphore = None
    
    @property
    def semaphore(self):
        """
        Lazy initialization of the semaphore.
        This ensures the semaphore is created in the event loop where it's used.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response with concurrency control"""
        async with self.semaphore:
            return await self.provider.generate(messages, **kwargs)
    
    def run_batch(self, 
                messages_list: List[List[Dict[str, str]]], 
                **kwargs) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, str]]]]:
        """Process batches with retries in separate event loops, using id() to track messages"""
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we get here, we're in an event loop, so use the async version
            import warnings
            warnings.warn("run_batch called from within event loop, consider using run_batch_async instead")
            return asyncio.get_event_loop().run_until_complete(self.run_batch_async(messages_list, **kwargs))
        except RuntimeError:
            # No event loop running, proceed with original implementation
            pass

        results = [None] * len(messages_list)
        position_map = {id(messages): i for i, messages in enumerate(messages_list)}
        
        # Queue to store unfinished or failed tasks
        current_batch = messages_list.copy()
        max_retries = kwargs.get("max_retries", 100)
        retry_count = 0
        
        while current_batch and retry_count < max_retries:
            async def process_batch():
                self._semaphore = None  # Reset semaphore for this event loop
                batch_results = []
                failures = []
                
                tasks_with_messages = [(msg, asyncio.create_task(self.generate(msg, **kwargs))) 
                                    for msg in current_batch]
                for messages, task in tasks_with_messages:
                    try:
                        response = await task
                        position = position_map[id(messages)]
                        batch_results.append((position, {
                            "messages": messages,
                            "response": response.content,
                            "model": response.model_name,
                            "success": True
                        }))
                    except Exception as e:
                        print(f'[DEBUG] error: {e}')
                        failures.append(messages)
                
                return batch_results, failures
            
            # Run in fresh event loop
            batch_results, next_batch = asyncio.run(process_batch())
            
            # Update results with successful responses
            for position, result in batch_results:
                results[position] = result
            
            # Update for next iteration
            if next_batch:
                retry_count += 1
                # Update position map for failed messages
                position_map = {id(messages): position_map[id(messages)] 
                            for messages in next_batch}
                
                current_batch = next_batch
                time.sleep(5)
                print(f'[DEBUG] {len(next_batch)} failed messages, retry_count: {retry_count}')
            else:
                break

        return results, next_batch
    
    async def run_batch_async(self, 
                            messages_list: List[List[Dict[str, str]]], 
                            **kwargs) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, str]]]]:
        """Async version of run_batch for use within event loops"""
        results = []
        failures = []
        
        # Create tasks for all messages
        tasks_with_messages = [(msg, asyncio.create_task(self.generate(msg, **kwargs))) 
                              for msg in messages_list]
        
        # Wait for all tasks to complete
        for messages, task in tasks_with_messages:
            try:
                response = await task
                results.append({
                    "messages": messages,
                    "response": response.content,
                    "model": response.model_name,
                    "success": True
                })
            except Exception as e:
                print(f'[DEBUG] error: {e}')
                failures.append(messages)
        
        return results, failures



if __name__ == "__main__":
    import sys
    
    # Test different providers
    if len(sys.argv) > 1 and sys.argv[1] == "local":
        # Test local VLLM provider (requires running VLLM server)
        base_url = "http://localhost:8000/v1"
        print(f"Testing local VLLM provider at {base_url}")
        llm = ConcurrentLLM(
            provider="vllm_local", 
            model_name="ragen_model",
            base_url=base_url,
            max_concurrency=4
        )
    else:
        # Test external providers
        # llm = ConcurrentLLM(provider="openai", model_name="gpt-4o")
        # llm = ConcurrentLLM(provider="anthropic", model_name="claude-3-5-sonnet-20240620")
        llm = ConcurrentLLM(provider="together", model_name="Qwen/Qwen2.5-7B-Instruct-Turbo")
    
    messages = [
        [{"role": "user", "content": "what is 2+2?"}],
        [{"role": "user", "content": "what is 2+3?"}],
        [{"role": "user", "content": "what is 2+4?"}],
        [{"role": "user", "content": "what is 2+5?"}],
        [{"role": "user", "content": "what is 2+6?"}],
        [{"role": "user", "content": "what is 2+7?"}],
        [{"role": "user", "content": "what is 2+8?"}],
        [{"role": "user", "content": "what is 2+9?"}],
        [{"role": "user", "content": "what is 2+10?"}],
        [{"role": "user", "content": "what is 2+11?"}],
        [{"role": "user", "content": "what is 2+12?"}],
        [{"role": "user", "content": "what is 2+13?"}],
        [{"role": "user", "content": "what is 2+14?"}],
        [{"role": "user", "content": "what is 2+15?"}],
        [{"role": "user", "content": "what is 2+16?"}],
        [{"role": "user", "content": "what is 2+17?"}],
    ]
    
    try:
        response = llm.run_batch(messages, max_tokens=100)
        print(f"✅ Test successful! Got {len(response[0])} responses")
        print(f"Sample response: {response[0][0]['response'][:100]}..." if response[0] else "No responses")
    except Exception as e:
        print(f"❌ Test failed: {e}")
