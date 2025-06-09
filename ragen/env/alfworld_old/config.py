from ragen.env.base import BaseEnvConfig
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class AlfredEnvConfig(BaseEnvConfig):
    """configuration for text world AlfredEnv"""
    config_file: str = "./ragen/env/alfworld_old/alfworld_config.yaml"
    format_score: float = 0.1
    score: float = 1.0
    render_mode: str = "text"