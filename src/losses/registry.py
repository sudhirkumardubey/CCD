"""
Loss model registry system from TurboFlow
"""

from typing import Dict, Callable, Any, List
from dataclasses import dataclass
import numpy as np


@dataclass
class LossContext:
    """Context for loss calculations"""
    component: str
    geometry: Any
    operating_condition: Any
    inlet_state: Any
    outlet_state:  Any
    velocity_triangle: Dict


class LossModelRegistry:
    """Central registry for all loss models"""
    
    _models: Dict[str, Dict[str, Callable]] = {}
    
    @classmethod
    def register(cls, component: str, model_name: str):
        """Decorator to register loss models"""
        def decorator(func:  Callable):
            if component not in cls._models:
                cls._models[component] = {}
            cls._models[component][model_name] = func
            func._registered = True
            func._component = component
            func._model_name = model_name
            return func
        return decorator
    
    @classmethod
    def get_model(cls, component: str, model_name:  str) -> Callable:
        """Retrieve a registered loss model"""
        if component not in cls._models:
            raise ValueError(f"No models for component '{component}'")
        if model_name not in cls._models[component]:
            available = list(cls._models[component].keys())
            raise ValueError(
                f"Model '{model_name}' not found for '{component}'. "
                f"Available:  {available}"
            )
        return cls._models[component][model_name]
    
    @classmethod
    def list_models(cls, component: str = None):
        """List all registered models"""
        if component: 
            return list(cls._models.get(component, {}).keys())
        return {c: list(m.keys()) for c, m in cls._models.items()}
    
    @classmethod
    def calculate_losses(cls, context: LossContext, model_config: Dict) -> Dict:
        """Calculate all losses for a component"""
        component = context.component
        losses = {}
        
        for loss_type, model_name in model_config.items():
            if component in cls._models and model_name in cls._models[component]:
                loss_func = cls._models[component][model_name]
                losses[loss_type] = loss_func(context)
        
        return losses