#!/usr/bin/env python3
"""
Simplified Embedding Cache Module
"""

import os
import pickle
import hashlib
import json
from typing import List, Dict, Any, Optional
from pathlib import Path


class EmbeddingCache:
    """Simple, reliable embedding cache."""
    
    def __init__(self, cache_dir: str = "embedding_cache", enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.memory_cache = {}
        
        if self.enabled:
            self.cache_dir.mkdir(exist_ok=True)
            print(f"📁 Cache initialized: {self.cache_dir}")
    
    def _make_key(self, texts: List[str], model: str) -> str:
        """Create a simple cache key."""
        # Use first few texts + model + count for deterministic key
        sample = texts[:3] if len(texts) > 3 else texts
        key_data = {
            'model': model,
            'count': len(texts),
            'sample': sample
        }
        content = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, texts: List[str], model: str) -> Optional[List[List[float]]]:
        """Get embeddings from cache."""
        if not self.enabled:
            return None
        
        key = self._make_key(texts, model)
        
        # Check memory first
        if key in self.memory_cache:
            print(f"  🔋 Memory cache hit for {model}")
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                self.memory_cache[key] = embeddings
                print(f"  💾 Disk cache hit for {model}")
                return embeddings
            except Exception as e:
                print(f"  ⚠️ Cache load failed: {e}")
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def put(self, texts: List[str], model: str, embeddings: List[List[float]]) -> None:
        """Store embeddings in cache."""
        if not self.enabled:
            return
        
        key = self._make_key(texts, model)
        
        # Store in memory
        self.memory_cache[key] = embeddings
        
        # Store on disk
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"  💾 Cached {len(embeddings)} embeddings for {model}")
        except Exception as e:
            print(f"  ⚠️ Cache save failed: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cache."""
        self.memory_cache.clear()
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()
        print("🗑️ Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "enabled": True,
            "memory_entries": len(self.memory_cache),
            "disk_entries": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024)
        }


    def get_by_params(self, model: str, dataset_type: str, n_samples: int, size: int, **kwargs) -> Optional[List[List[float]]]:
        """Get embeddings by dataset parameters instead of actual texts."""
        if not self.enabled:
            return None
        
        # Create parameter-based key
        params = {
            'model': model,
            'dataset_type': dataset_type,
            'n_samples': n_samples,
            'size': size,
            'random_state': kwargs.get('random_state', 42),
            **kwargs
        }
        content = json.dumps(params, sort_keys=True)
        key = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Check memory first
        if key in self.memory_cache:
            print(f"  🔋 Parameter cache hit for {model}")
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"params_{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                self.memory_cache[key] = embeddings
                print(f"  💾 Parameter cache hit for {model}")
                return embeddings
            except Exception as e:
                print(f"  ⚠️ Parameter cache load failed: {e}")
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def put_by_params(self, embeddings: List[List[float]], model: str, dataset_type: str, 
                      n_samples: int, size: int, **kwargs) -> None:
        """Store embeddings by dataset parameters."""
        if not self.enabled:
            return
        
        # Create parameter-based key
        params = {
            'model': model,
            'dataset_type': dataset_type,
            'n_samples': n_samples,
            'size': size,
            'random_state': kwargs.get('random_state', 42),
            **kwargs
        }
        content = json.dumps(params, sort_keys=True)
        key = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Store in memory
        self.memory_cache[key] = embeddings
        
        # Store on disk
        cache_file = self.cache_dir / f"params_{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"  💾 Parameter cached for {model}")
        except Exception as e:
            print(f"  ⚠️ Parameter cache save failed: {e}")


class CachedEmbeddingWrapper:
    """Wrapper that adds caching to any embedding wrapper."""
    
    def __init__(self, embedding_wrapper, cache_dir: str = "embedding_cache", 
                 enable_cache: bool = True):
        self.wrapper = embedding_wrapper
        self.cache = EmbeddingCache(cache_dir, enable_cache)
    
    def __getattr__(self, name):
        """Forward all other attributes to the wrapped object."""
        return getattr(self.wrapper, name)
    
    def embed(self, texts: List[str], model: str) -> List[List[float]]:
        """Get embeddings with caching."""
        # Try cache first
        cached = self.cache.get(texts, model)
        if cached is not None:
            return cached
        
        # Get from API
        print(f"  🌐 API call for {model}")
        embeddings = self.wrapper.embed(texts, model)
        
        # Cache it
        self.cache.put(texts, model, embeddings)
        
        return embeddings
    
    def embed_with_params(self, texts: List[str], model: str, dataset_type: str, 
                          size: int, **kwargs) -> List[List[float]]:
        """Get embeddings with parameter-based caching for reproducible datasets."""
        # Try parameter cache first
        cached = self.cache.get_by_params(
            model, dataset_type, len(texts), size, **kwargs
        )
        if cached is not None:
            return cached
        
        # Try regular cache as fallback
        cached = self.cache.get(texts, model)
        if cached is not None:
            print(f"  🔋 Text cache hit for {model}")
            # Also store in parameter cache for next time
            self.cache.put_by_params(cached, model, dataset_type, len(texts), size, **kwargs)
            return cached
        
        # Get from API
        print(f"  🌐 API call for {model}")
        embeddings = self.wrapper.embed(texts, model)
        
        # Cache both ways
        self.cache.put(texts, model, embeddings)
        self.cache.put_by_params(embeddings, model, dataset_type, len(texts), size, **kwargs)
        
        return embeddings
    
    def preload_embeddings(self, texts: List[str], models: List[str]) -> Dict[str, bool]:
        """Preload embeddings for multiple models."""
        results = {}
        print(f"📦 Preloading embeddings for {len(models)} models...")
        
        for model in models:
            try:
                # Check if already cached
                if self.cache.get(texts, model) is not None:
                    print(f"  ✓ {model} (already cached)")
                    results[model] = True
                    continue
                
                # Fetch and cache
                self.embed(texts, model)
                results[model] = True
                print(f"  ✓ {model} (preloaded)")
                
            except Exception as e:
                print(f"  ✗ {model} (failed: {e})")
                results[model] = False
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()