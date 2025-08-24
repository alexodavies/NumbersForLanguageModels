"""
Universal Embeddings API Wrapper

A unified interface for OpenAI, Google Gemini, and Voyage AI embedding models.
Automatically detects which service to use based on the model name.
"""

import os
from typing import List, Union, Optional, Dict, Any
import warnings

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import voyageai
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class EmbeddingWrapper:
    """
    Universal wrapper for embedding APIs from OpenAI, Google Gemini, and Voyage AI.
    
    Supported models:
    - OpenAI: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    - Google Gemini: gemini-embedding-001, gemini-embedding-exp-03-07
    - Voyage AI: voyage-3-large, voyage-3.5, voyage-3.5-lite, voyage-code-3, 
                 voyage-finance-2, voyage-law-2, voyage-multimodal-3
    """
    
    # Model mappings to determine which service to use
    OPENAI_MODELS = {
        'text-embedding-3-small',
        'text-embedding-3-large', 
        'text-embedding-ada-002'
    }
    
    GOOGLE_MODELS = {
        'gemini-embedding-001'
        # 'gemini-embedding-exp-03-07'
    }
    
    VOYAGE_MODELS = {
        'voyage-3-large',
        'voyage-3.5',
        'voyage-3.5-lite', 
        'voyage-code-3',
        'voyage-finance-2',
        'voyage-law-2',
        'voyage-multimodal-3'
    }
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 google_api_key: Optional[str] = None,
                 voyage_api_key: Optional[str] = None):
        """
        Initialize the wrapper with API keys.
        
        Args:
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            google_api_key: Google API key (or set GOOGLE_API_KEY env var)  
            voyage_api_key: Voyage AI API key (or set VOYAGE_API_KEY env var)
        """
        # Initialize clients
        self.openai_client = None
        self.google_client = None
        self.voyage_client = None
        
        # Setup OpenAI
        if OPENAI_AVAILABLE:
            api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
        
        # Setup Google Gemini
        if GOOGLE_AVAILABLE:
            api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
            if api_key:
                os.environ['GOOGLE_API_KEY'] = api_key
                try:
                    self.google_client = genai.Client()
                except Exception:
                    # Fallback: try creating client with API key parameter
                    try:
                        self.google_client = genai.Client(api_key=api_key)
                    except Exception:
                        warnings.warn("Failed to initialize Google Gemini client")
            elif os.getenv('GOOGLE_API_KEY'):
                try:
                    self.google_client = genai.Client()
                except Exception:
                    warnings.warn("Failed to initialize Google Gemini client with env var")
        
        # Setup Voyage AI
        if VOYAGE_AVAILABLE:
            api_key = voyage_api_key or os.getenv('VOYAGE_API_KEY')
            if api_key:
                self.voyage_client = voyageai.Client(api_key=api_key)
    
    def _detect_service(self, model: str) -> str:
        """Detect which service to use based on model name."""
        if model in self.OPENAI_MODELS:
            return 'openai'
        elif model in self.GOOGLE_MODELS:
            return 'google'
        elif model in self.VOYAGE_MODELS:
            return 'voyage'
        else:
            raise ValueError(f"Unknown model: {model}. Supported models: "
                           f"OpenAI: {self.OPENAI_MODELS}, "
                           f"Google: {self.GOOGLE_MODELS}, "
                           f"Voyage: {self.VOYAGE_MODELS}")
    
    def _openai_embed(self, 
                      text: Union[str, List[str]], 
                      model: str,
                      **kwargs) -> List[List[float]]:
        """Get embeddings from OpenAI."""
        if not self.openai_client:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not available. Install with: pip install openai")
            else:
                warnings.warn("OpenAI API key not provided. Using mock data for testing.")
                # Return mock embeddings for testing
                texts = [text] if isinstance(text, str) else text
                dimensions = kwargs.get('dimensions')
                dim = dimensions or (1536 if model == 'text-embedding-3-small' else 3072)
                return [[0.0] * dim for _ in texts]
        
        # Prepare parameters
        params = {'input': text, 'model': model}
        if 'dimensions' in kwargs:
            params['dimensions'] = kwargs['dimensions']
            
        response = self.openai_client.embeddings.create(**params)
        return [item.embedding for item in response.data]
    
    def _google_embed(self, 
                    text: Union[str, List[str]], 
                    model: str,
                    **kwargs) -> List[List[float]]:
        """Get embeddings from Google Gemini with proper batching."""
        if not self.google_client:
            if not GOOGLE_AVAILABLE:
                raise ImportError("Google genai package not available. Install with: pip install google-genai")
            else:
                warnings.warn("Google API key not provided. Using mock data for testing.")
                # Return mock embeddings for testing
                texts = [text] if isinstance(text, str) else text
                output_dimensionality = kwargs.get('output_dimensionality')
                dim = output_dimensionality or 3072
                return [[0.0] * dim for _ in texts]
        
        # Convert single string to list for consistent processing
        texts = [text] if isinstance(text, str) else text
        
        # Prepare config
        task_type = kwargs.get('task_type', 'SEMANTIC_SIMILARITY')
        config_params = {'task_type': task_type}
        if 'output_dimensionality' in kwargs:
            config_params['output_dimensionality'] = kwargs['output_dimensionality']
        
        config = types.EmbedContentConfig(**config_params)
        
        # Google's batch limit is 100 requests
        GOOGLE_BATCH_SIZE = 100
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), GOOGLE_BATCH_SIZE):
            batch = texts[i:i + GOOGLE_BATCH_SIZE]
            
            # Add small delay between batches to respect rate limits
            if i > 0:
                import time
                time.sleep(0.1)
            
            try:
                result = self.google_client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=config
                )
                
                batch_embeddings = [list(emb.values) for emb in result.embeddings]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                # If batch fails, try one by one (for debugging)
                print(f"Batch failed with {len(batch)} items, trying individually: {e}")
                for single_text in batch:
                    try:
                        single_result = self.google_client.models.embed_content(
                            model=model,
                            contents=[single_text],
                            config=config
                        )
                        all_embeddings.extend([list(emb.values) for emb in single_result.embeddings])
                    except Exception as single_e:
                        print(f"Failed on single text: {single_e}")
                        raise single_e
        
        return all_embeddings
    
    def _voyage_embed(self, 
                      text: Union[str, List[str]], 
                      model: str,
                      **kwargs) -> List[List[float]]:
        """Get embeddings from Voyage AI."""
        if not self.voyage_client:
            if not VOYAGE_AVAILABLE:
                raise ImportError("Voyage AI package not available. Install with: pip install voyageai")
            else:
                warnings.warn("Voyage AI API key not provided. Using mock data for testing.")
                # Return mock embeddings for testing
                texts = [text] if isinstance(text, str) else text
                return [[0.0] * 1024 for _ in texts]  # Voyage default is 1024
        
        # Convert single string to list for consistent processing
        texts = [text] if isinstance(text, str) else text
        input_type = kwargs.get('input_type', 'document')
        
        result = self.voyage_client.embed(
            texts, 
            model=model, 
            input_type=input_type
        )
        
        return result.embeddings
    
    def embed(self, 
              text: Union[str, List[str]], 
              model: str,
              **kwargs) -> Union[List[float], List[List[float]]]:
        """
        Get embeddings for text using the specified model.
        
        Args:
            text: String or list of strings to embed
            model: Model name (determines which service to use)
            **kwargs: Service-specific parameters:
                - OpenAI: dimensions
                - Google: task_type, output_dimensionality  
                - Voyage: input_type
        
        Returns:
            List of floats (single text) or List of Lists of floats (multiple texts)
        """
        # Input validation
        if not text:
            raise ValueError("Text input cannot be empty")
        
        if isinstance(text, list) and len(text) == 0:
            raise ValueError("Text list cannot be empty")
        
        if isinstance(text, list) and any(not t or not isinstance(t, str) for t in text):
            raise ValueError("All texts in list must be non-empty strings")
        
        service = self._detect_service(model)
        was_single = isinstance(text, str)
        
        try:
            if service == 'openai':
                embeddings = self._openai_embed(text, model, **kwargs)
            elif service == 'google':
                embeddings = self._google_embed(text, model, **kwargs)
            elif service == 'voyage':
                embeddings = self._voyage_embed(text, model, **kwargs)
            else:
                raise ValueError(f"Unsupported service: {service}")
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Error getting embeddings from {service} with model {model}: {str(e)}") from e
        
        # Return single embedding if input was single string
        return embeddings[0] if was_single else embeddings
    
    def get_supported_models(self) -> Dict[str, List[str]]:
        """Get all supported models grouped by service."""
        return {
            'openai': list(self.OPENAI_MODELS),
            'google': list(self.GOOGLE_MODELS), 
            'voyage': list(self.VOYAGE_MODELS)
        }
    
    def get_available_services(self) -> Dict[str, bool]:
        """Check which services are available (have packages installed and API keys configured)."""
        return {
            'openai': OPENAI_AVAILABLE and self.openai_client is not None,
            'google': GOOGLE_AVAILABLE and self.google_client is not None,
            'voyage': VOYAGE_AVAILABLE and self.voyage_client is not None
        }


# Convenience function for quick usage
def get_embedding(text: Union[str, List[str]], 
                  model: str,
                  **kwargs) -> Union[List[float], List[List[float]]]:
    """
    Convenience function to get embeddings without creating a wrapper instance.
    
    Args:
        text: String or list of strings to embed
        model: Model name
        **kwargs: Service-specific parameters
    
    Returns:
        Embedding vector(s)
    """
    wrapper = EmbeddingWrapper()
    return wrapper.embed(text, model, **kwargs)


# Example usage
if __name__ == "__main__":
    # Initialize wrapper
    wrapper = EmbeddingWrapper()
    
    # Example texts
    single_text = "What is the meaning of life?"
    multiple_texts = [
        "What is the meaning of life?",
        "How do I bake a cake?",
        "What is machine learning?"
    ]
    
    print("Supported models:")
    for service, models in wrapper.get_supported_models().items():
        print(f"  {service}: {models}")
    
    print("\nService availability:")
    for service, available in wrapper.get_available_services().items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {service}: {status}")
    
    print("\n" + "="*50)
    print("Testing different models (using mock data):")
    
    # Test OpenAI
    print("\n1. OpenAI text-embedding-3-small:")
    try:
        embedding = wrapper.embed(single_text, "text-embedding-3-small")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test Google Gemini
    print("\n2. Google gemini-embedding-001:")
    try:
        embedding = wrapper.embed(single_text, "gemini-embedding-001", 
                                task_type="SEMANTIC_SIMILARITY")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test Voyage AI
    print("\n3. Voyage AI voyage-3.5:")
    try:
        embedding = wrapper.embed(single_text, "voyage-3.5", input_type="query")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test multiple texts
    print("\n4. Multiple texts with OpenAI:")
    try:
        embeddings = wrapper.embed(multiple_texts, "text-embedding-3-small")
        print(f"   Number of embeddings: {len(embeddings)}")
        print(f"   Each embedding dimension: {len(embeddings[0])}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test convenience function
    print("\n5. Using convenience function:")
    try:
        embedding = get_embedding("Hello world!", "text-embedding-3-small")
        print(f"   Embedding dimension: {len(embedding)}")
    except Exception as e:
        print(f"   Error: {e}")