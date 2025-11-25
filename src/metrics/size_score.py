from typing import Any, Dict

from ..models import MetricResult, ModelContext, SizeScore
from ..utils import extract_model_size_from_text, measure_time
from .base import BaseMetric
import math
import re


class SizeScoreMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "size_score"

    def compute(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> MetricResult:
        # compute size score for different hardware targets
        with measure_time() as get_latency:
            size_score = self._calculate_size_scores(context, config)
        
        # calculate average score across all devices
        avg_score = (
            size_score.raspberry_pi +
            size_score.jetson_nano +
            size_score.desktop_pc +
            size_score.aws_server
        ) / 4.0
        
        # return MetricResult with average float score
        return MetricResult(
            score=avg_score,
            latency=get_latency()
        )

    def _calculate_size_scores(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> SizeScore:
        # estimate model size from various sources
        estimated_size_gb = self._estimate_model_size(context)

        # get thresholds from config, with fallbacks
        size_limits = config.get("thresholds", {}).get("size_limits", {})
        softness = float(config.get("thresholds",{}).get("softness", 1.2))
        
        raspberry_pi_limit = size_limits.get("raspberry_pi", 2.0)
        jetson_nano_limit = size_limits.get("jetson_nano", 8.0)
        desktop_pc_limit = size_limits.get("desktop_pc", 32.0)
        aws_server_limit = size_limits.get("aws_server", 128.0)

        return SizeScore(
            raspberry_pi=self._calculate_device_score(estimated_size_gb, raspberry_pi_limit),
            jetson_nano=self._calculate_device_score(estimated_size_gb, jetson_nano_limit),
            desktop_pc=self._calculate_device_score(estimated_size_gb, desktop_pc_limit),
            aws_server=self._calculate_device_score(estimated_size_gb, aws_server_limit),
        )

    def _calculate_device_score(self, model_size_gb: float, limit_gb: float) -> float:
        """ calculate normalized score versus device limit:
        
        - <= limit - 1.0
        - <= 2x limit - 0.8  
        - <= 5x limit - 0.5
        - > 5x limit - 0.0
        """
        if limit_gb <= 0:
            return 0.0
        
        ratio = model_size_gb / limit_gb

        softness = 1.2  # tweak
        score = 1.0 / (1.0 + math.pow(ratio, softness))
        return round(score, 3)

    def _estimate_model_size(self, context: ModelContext) -> float:
        """Estimate model size from various sources."""
        
        # try using existing utility function
        try:
            # different ways to get text for the utility function
            text_to_analyze = ""
            if hasattr(context, 'model_url') and context.model_url:
                if hasattr(context.model_url, 'name'):
                    text_to_analyze = context.model_url.name
                else:
                    text_to_analyze = str(context.model_url)
                    
                size_from_text = extract_model_size_from_text(text_to_analyze)
                if size_from_text and size_from_text > 0:
                    return size_from_text
        except Exception:
            pass  # fall back to manual parsing
        
        # try HuggingFace file info
        if hasattr(context, 'hf_info') and context.hf_info and context.hf_info.get("files"):
            total_size_gb = 0.0
            model_files = 0
            
            for file_path in context.hf_info["files"]:
                if file_path.endswith((".bin", ".safetensors")):
                    # check for file size info 
                    file_info = context.hf_info.get("file_info", {}).get(file_path, {})
                    if "size" in file_info:
                        total_size_gb += file_info["size"] / (1024**3)
                    else:
                        # conservative estimate for model files
                        total_size_gb += 0.25
                    model_files += 1
                elif file_path.endswith(".h5"):
                    total_size_gb += 0.8
                    model_files += 1
                elif file_path.endswith((".json", ".txt", ".md", ".py", ".gitignore")):
                    total_size_gb += 0.001  # config files
                    
            if model_files > 0:
                return max(total_size_gb, 0.01)
        
        # extract from model name/URL
        model_name = ""
        if hasattr(context, 'model_url') and context.model_url:
            if hasattr(context.model_url, 'name'):
                model_name = context.model_url.name.lower()
            else:
                model_name = str(context.model_url).lower()
        
        if not model_name:
            return 0.5  # default fallback
        
        # model size mappings for bert, whisper, audience, gemma
        model_size_mappings = {
            'bert-base-uncased': 0.44,    # ~110M parameters
            'whisper-tiny': 0.075,        # ~39M parameters  
            'audience-classifier': 0.1,   # estimated small classifier
            'gemma-3-270m': 0.54, 
        }
        
        # check for exact model matches first
        model_name_clean = model_name.lower().replace('_', '-').replace(' ', '-')
        for model_key, size in model_size_mappings.items():
            if model_key in model_name_clean:
                return size
        
        # billion parameters
        b_patterns = [
            r'(\d+(?:\.\d+)?)b(?:-|_|$|\s)',  # 7b, 1.3b followed by delimiter or end
            r'(\d+(?:\.\d+)?)-?billion',      # 7billion, 7-billion
        ]
        
        for pattern in b_patterns:
            match = re.search(pattern, model_name)
            if match:
                param_count = float(match.group(1))
                #sStandard estimate: 2GB per billion parameters for 16-bit models
                return param_count * 2.0
        
        # million parameters  
        m_patterns = [
            r'(\d+(?:\.\d+)?)m(?:-|_|$|\s)',  # 125m, 350m
            r'(\d+(?:\.\d+)?)-?million',      # 125million, 125-million
        ]
        
        for pattern in m_patterns:
            match = re.search(pattern, model_name)
            if match:
                param_count = float(match.group(1))
                # 2MB per million parameters
                return param_count * 0.002
        
        # direct size patterns
        gb_patterns = [
            r'(\d+(?:\.\d+)?)gb',
            r'(\d+(?:\.\d+)?)g(?:-|_|$|\s)',
        ]
        
        for pattern in gb_patterns:
            match = re.search(pattern, model_name)
            if match:
                return float(match.group(1))
        
        # model family/architecture-specific heuristics -more accurate estimates
        architecture_sizes = {
            # BERT family
            ('bert-large',): 1.3,        # ~340M params
            ('bert-base',): 0.44,        # ~110M params  
            ('distilbert',): 0.26,       # ~66M params
            
            # whisper family
            ('whisper-tiny',): 0.075,    # ~39M params
            ('whisper-small',): 0.24,    # ~61M params
            ('whisper-base',): 0.29,     # ~74M params
            ('whisper-medium',): 1.53,   # ~769M params
            ('whisper-large',): 3.09,    # ~1550M params
            
            # T5 family
            ('t5-small',): 0.24,         # ~60M params
            ('t5-base',): 0.89,          # ~220M params
            ('t5-large',): 3.0,          # ~770M params
            
            # GPT family
            ('gpt2',): 0.5,              # ~117M params
            ('gpt2-medium',): 1.4,       # ~345M params
            ('gpt2-large',): 3.2,        # ~774M params
            
            # generic size indicators (fallbacks)
            ('mini', 'tiny', 'nano'): 0.1,
            ('small',): 0.3,
            ('base', 'medium'): 0.8,
            ('large', 'big'): 2.5,
            ('xl', 'extra-large'): 4.0,
            ('xxl', 'ultra', 'giant'): 12.0,
        }
        
        for keywords, size in architecture_sizes.items():
            if any(keyword in model_name for keyword in keywords):
                return size
        
        # default fallback
        return 0.5