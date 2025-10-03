import requests
import json
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

# Import path fix for local runs
try:
    from config import Config  # noqa: F401 (kept for potential future use)
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import Config  # noqa: F401

class LLMHandler:
    """Handles interactions with Language Models"""
    
    def __init__(self, model_type: str, model_name: str, api_key: str = "", api_base: str = "", max_tokens: int = 1000):
        """
        Initialize LLM handler
        
        Args:
            model_type: Type of model to use ("openai" or "simple")
                        "openai" is used for OpenRouter
            model_name: Name of the model to use ("deepseek/deepseek-chat:free")
            api_key: API key for OpenAI/OpenRouter
            api_base: Base URL for OpenAI-compatible API (OpenRouter)
            max_tokens: Maximum tokens for LLM response
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM based on model type
        if self.model_type == "openai":
            self.api_key = api_key
            self.api_base = api_base
            
            if not self.api_key:
                self.logger.warning("OpenAI API key not found, falling back to simple mode.")
                self.model_type = "simple"
            else:
                self.logger.info(f"Initialized OpenAI/OpenRouter LLM: {self.model_name} (API Base: {self.api_base})")
        elif self.model_type == "simple":
            self.logger.info("Initialized Simple Rule-based LLM.")
        else:
            self.logger.warning(f"Unsupported LLM type '{model_type}', falling back to simple mode.")
            self.model_type = "simple"

    def generate_response(self, prompt: str, context: Optional[str] = None, relevant_content: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a response using the configured LLM.
        
        Args:
            prompt: User's query.
            context: The retrieved relevant documents (prepared string).
            relevant_content: List of raw content strings from relevant documents (used by simple mode).
            
        Returns:
            Dictionary containing the response, model used, and token count.
        """
        if self.model_type == "openai":
            return self._generate_openai_response(prompt, context)
        elif self.model_type == "simple":
            return self._generate_simple_fallback(prompt, relevant_content)
        else:
            return {
                'response': "LLM not configured correctly. Please check settings.",
                'model_used': "N/A",
                'tokens_used': 0,
                'error': True
            }

    def _generate_openai_response(self, prompt: str, context: Optional[str]) -> Dict[str, Any]:
        """Generate response using OpenAI-compatible API (OpenRouter)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [{"role": "system", "content": "You are an AI assistant specialized in providing accurate information about disaster preparedness based on the provided context. Answer the user's question only using the provided context. If the answer is not in the context, state that you cannot answer based on the provided documents and suggest seeking official emergency information."}]
        
        if context:
            messages.append({"role": "user", "content": f"Based on the following information, answer the question:\n\nContext: {context}\n\nQuestion: {prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})
            
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens, # Use instance's max_tokens
            "temperature": 0.0 # Temperature is low for factual RAG responses
        }
        
        try:
            self.logger.info(f"Calling OpenAI-compatible API ({self.api_base}) for model: {self.model_name}...")
            response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses
            
            response_json = response.json()
            
            if response_json.get("choices") and response_json["choices"][0].get("message"):
                llm_response = response_json["choices"][0]["message"]["content"]
                tokens_used = response_json.get("usage", {}).get("total_tokens", 0)
                
                return {
                    'response': llm_response,
                    'model_used': self.model_name,
                    'tokens_used': tokens_used,
                    'error': False
                }
            else:
                self.logger.error(f"Invalid response from OpenAI API: {response_json}")
                return {
                    'response': "Error: Invalid response from LLM.",
                    'model_used': self.model_name,
                    'tokens_used': 0,
                    'error': True
                }
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network or API error calling OpenAI: {str(e)}")
            return {
                'response': f"Network or API error: {str(e)}",
                'model_used': self.model_name,
                'tokens_used': 0,
                'error': True
            }
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error from OpenAI API: {str(e)}. Response: {response.text}")
            return {
                'response': f"Error parsing LLM response: {str(e)}",
                'model_used': self.model_name,
                'tokens_used': 0,
                'error': True
            }
        except Exception as e:
            self.logger.error(f"Unexpected error in _generate_openai_response: {str(e)}")
            return {
                'response': f"An unexpected error occurred: {str(e)}",
                'model_used': self.model_name,
                'tokens_used': 0,
                'error': True
            }

    def _generate_simple_fallback(self, prompt: str, relevant_content: Optional[List[str]]) -> Dict[str, Any]:
        """Generates a simple, rule-based response when no LLM is available."""
        self.logger.info("Generating simple response...")
        response = ""
        try:
            if "hello" in prompt.lower() or "hi" in prompt.lower():
                response = "Hello! I am a chatbot designed to help you with disaster preparedness information. How can I assist you today?"
            elif "thank" in prompt.lower():
                response = "You're welcome! Feel free to ask if you have more questions."
            elif not relevant_content:
                response = ("I couldn't find specific information related to your question in the knowledge base. "
                            "Please try asking something more general about disaster preparedness or rephrase your question.")
            else:
                # Create a simple response based on the context
                response = (f"Based on the disaster preparedness documents, here's what I found:\\n\\n"\
                           f"{' '.join(relevant_content[:3])}\\n\\n"
                           f"This information comes from the uploaded disaster preparedness guides.")
            
            # Limit response length
            if len(response) > self.max_tokens: # Use instance's max_tokens
                response = response[:self.max_tokens] + "..."
            
            return {
                'response': response,
                'model_used': f"simple:{self.model_name}",
                'tokens_used': len(response),
                'error': False
            }
            
        except Exception as e:
            self.logger.error(f"Simple generation failed: {str(e)}") # Log error before re-raising
            raise Exception(f"Simple generation failed: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        api_endpoint = "Simple Rule-based"
        if self.model_type == "openai":
            api_endpoint = self.api_base
        
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'api_endpoint': api_endpoint,
            'max_tokens': self.max_tokens
        }
