import argparse
import os
import yaml
import re
from typing import List, Any
import tempfile

def load_config(config_file: str, params: List[str] = []):
    assert os.path.exists(config_file), f"Invalid config file: {config_file}"
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
    # Parse overriden params.
    for param in params:
        fqn_key, value = param.split("=")
        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = value
    return config

def check_format(action: str, templates: Any) -> bool:
    """
    Validate that the action matches one of our action templates.
    Returns True if valid, False otherwise.
    """
    if "None" in action:
        return False
    
    # Skip validation for basic actions that don't have placeholders
    basic_actions = ["look", "inventory"]
    if action in basic_actions:
        return True
        
    # Check if the action follows any of our templates
    for template in templates:
        # Skip "None" and basic actions we already checked
        if template == "None" or template in basic_actions:
            continue
            
        # Convert template to regex pattern
        # Replace <something> with regex that matches any word(s)
        pattern = template.replace("<receptacle>", "([\\w\\s]+)") \
                        .replace("<object>", "([\\w\\s]+)") \
                        .replace("<something>", "([\\w\\s]+)")
        pattern = f"^{pattern}$"  # Match the entire string
        
        if re.match(pattern, action):
            return True
            
    return False

def check_correctness(action: str, target: str) -> bool:
    ...


def handle_temp_dir():
    # check exists and have write permission
    if os.path.exists('/scratch') and os.access('/scratch', os.W_OK):
        os.makedirs('/scratch/zengyif/tmp', exist_ok=True)
        tempfile.tempdir = '/scratch/zengyif/tmp'
    elif os.path.exists('/raid') and os.access('/raid', os.W_OK):
        os.makedirs('/raid/zengyif/tmp', exist_ok=True)
        tempfile.tempdir = '/raid/zengyif/tmp'
    else:
        print("Warning: No scratch or raid directory found. Using default temp directory")