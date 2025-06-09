"""
This is the environment for the ALFRED dataset.
author: Qineng Wang
date: 2025-03-30
"""
import random
import textworld
import textworld.gym
import numpy as np
import os
import glob
import alfworld.agents.modules.generic as generic
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv, AlfredDemangler, AlfredInfos
from ragen.env.base import BaseLanguageBasedEnv
from .config import AlfredEnvConfig
from .utils import load_config, check_format


class AlfredTXTEnv(BaseLanguageBasedEnv):

    def __init__(self, config: AlfredEnvConfig = AlfredEnvConfig()):
        super().__init__()
        self.config = config
        
        # Assign a specific game file to this environment instance
        # This will be set by the environment manager during initialization
        self.assigned_game_file = None
        
        # Current environment state
        self.current_env = None
        self.render_cache = None
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
    
    def assign_game_file(self, game_file_path: str):
        """
        Assign a specific game file to this environment instance.
        This should be called during environment initialization.
        
        Args:
            game_file_path: Full path to the game file to assign
        """
        if not game_file_path or not os.path.exists(game_file_path):
            raise ValueError(f"Game file does not exist: {game_file_path}")
        
        self.assigned_game_file = game_file_path
        # print(f"Assigned game file: {self.assigned_game_file.split('/')[-3:]}")
        
    def _get_all_game_files(self):
        """Get all available game files without loading them into memory."""
        try:
            from alfworld.info import ALFWORLD_DATA
            data_path = os.path.expandvars(ALFWORLD_DATA)
            
            # Task type mapping
            task_types = {
                1: "pick_and_place_simple",
                2: "look_at_obj_in_light", 
                3: "pick_clean_then_place_in_recep",
                4: "pick_heat_then_place_in_recep",
                5: "pick_cool_then_place_in_recep",
                6: "pick_two_obj_and_place"
            }
            
            # Find all game files
            all_game_files = []
            for task_id, task_name in task_types.items():
                search_pattern = f"{data_path}/json_2.1.1/train/{task_name}-*/*/game.tw-pddl"
                all_game_files.extend(list(glob.glob(search_pattern)))
            
            return all_game_files
        except Exception as e:
            print(f"Warning: Could not load ALFWORLD_DATA games: {e}")
            # Fallback to using a minimal raw_env for getting game files
            raw_env_config = load_config(self.config.config_file)
            temp_env = AlfredTWEnv(config=raw_env_config, train_eval="train")
            game_files = temp_env.game_files
            temp_env.close()
            return game_files
    
    def reset(self, seed=None, mode=None):
        """
        Reset the environment.
        For assigned game file mode, always uses the pre-assigned game file.
        For compatibility with existing code, seed can still be used for randomness within the game.
        """
        # Use assigned game file if available, otherwise fall back to seed-based selection
        if self.assigned_game_file is not None:
            selected_game = self.assigned_game_file
        else:
            # Fallback to original behavior for backward compatibility - get game files on demand
            all_game_files = self._get_all_game_files()
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)
                game_idx = seed % len(all_game_files)
                selected_game = all_game_files[game_idx]
            else:
                selected_game = random.choice(all_game_files)

        # Close previous environment if it exists
        if hasattr(self, 'alfred_env') and self.alfred_env is not None:
            self.alfred_env.close()
        if self.current_env is not None:
            self.current_env.close()

        # Use direct TextWorld registration instead of AlfredTWEnv for speed
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        config = load_config(self.config.config_file)
        wrappers = [AlfredDemangler(), AlfredInfos()]
        max_steps = config["rl"]["training"]["max_nb_steps_per_episode"]

        env_id = textworld.gym.register_game(
            selected_game,
            request_infos=request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=max_steps,
            wrappers=wrappers
        )

        self.alfred_env = textworld.gym.make(env_id)

        obs, info = self.alfred_env.reset()
        # Remove the TextWorld welcome message for cleaner output
        observation = obs[0].replace("-= Welcome to TextWorld, ALFRED! =-\n\n", "")
        self.render_cache = observation
        return self.render_cache

    
    def compute_score(self, base_reward, done):
        """
        Compute the score based on the base reward, format reward, and completion status.
        
        Args:
            base_reward: The reward from the environment
            done: Whether the episode is finished
            
        Returns:
            The computed score
        """
        if done:
            return self.config.score + self.config.format_score + base_reward
        else:
            return base_reward + self.config.format_score
    
    def step(self, action: str):
        """
        Take a step in the environment using the provided action string.
        The action must match one of the templates in ACTION_LOOKUP.
        """
        obs, rewards, dones, infos = self.alfred_env.step([action])  # BatchEnv expects a list of commands
        
        observation = obs[0]
        self.render_cache = observation

        # If "nothing happened", then the action is invalid
        if "nothing happens" in observation.lower():
            return f"Invalid action format: {action}", 0, False, {"action_is_effective": False, "action_is_valid": False, "success": False}
    
        base_reward = rewards[0]
        done = dones[0]
        info = {"action_is_effective": True, "action_is_valid": True, "success": done}
        
        reward = self.compute_score(base_reward, done)
        
        return self.render_cache, reward, done, info
    
    def render(self):
        return self.render_cache
    
    def close(self):
        self.render_cache = None
        if hasattr(self, 'alfred_env') and self.alfred_env is not None:
            self.alfred_env.close()
        if self.current_env is not None:
            self.current_env.close()


def get_all_alfworld_game_files(config_file=None):
    """
    Static method to get all game files without creating an environment instance.
    This is used by the environment manager to distribute games to environments.
    """
    try:
        from alfworld.info import ALFWORLD_DATA
        data_path = os.path.expandvars(ALFWORLD_DATA)
        
        # Task type mapping
        task_types = {
            1: "pick_and_place_simple",
            2: "look_at_obj_in_light", 
            3: "pick_clean_then_place_in_recep",
            4: "pick_heat_then_place_in_recep",
            5: "pick_cool_then_place_in_recep",
            6: "pick_two_obj_and_place"
        }
        
        # Find all game files
        all_game_files = []
        for task_id, task_name in task_types.items():
            search_pattern = f"{data_path}/json_2.1.1/train/{task_name}-*/*/game.tw-pddl"
            all_game_files.extend(list(glob.glob(search_pattern)))
        
        return all_game_files
    except Exception as e:
        print(f"Warning: Could not load ALFWORLD_DATA games: {e}")
        # Fallback to using a minimal raw_env for getting game files
        if config_file is None:
            config_file = "./ragen/env/alfworld_old/alfworld_config.yaml"
        raw_env_config = load_config(config_file)
        temp_env = AlfredTWEnv(config=raw_env_config, train_eval="train")
        game_files = temp_env.game_files
        temp_env.close()
        return game_files


if __name__ == "__main__":
    # Test single game assignment
    config = AlfredEnvConfig()
    
    print("Testing Alfworld single game assignment...")
    
    # Create environment
    env = AlfredTXTEnv(config)
    
    # Get all game files for testing
    all_game_files = get_all_alfworld_game_files(config.config_file)
    print(f"Total game files available: {len(all_game_files)}")
    
    # Test with assigned game file
    if all_game_files:
        test_game_file = all_game_files[42 % len(all_game_files)]  # Assign a specific game
        env.assign_game_file(test_game_file)
        
        print("\n=== Testing with assigned game file ===")
        obs1 = env.reset()
        print(f"Reset 1 - Game: {env.assigned_game_file.split('/')[-3:]}")
        print(f"Observation: {obs1[:100]}...")
        
        # Reset again - should use same game
        obs2 = env.reset()
        print(f"Reset 2 - Game: {env.assigned_game_file.split('/')[-3:]}")
        print(f"Same observation: {obs1 == obs2}")
        
        # Test a simple action
        action = "look"
        obs, reward, done, info = env.step(action)
        print(f"After '{action}': reward={reward}, done={done}")
        
        env.close()
        
        # Test with different game assignment
        print("\n=== Testing with different game assignment ===")
        env2 = AlfredTXTEnv(config)
        test_game_file2 = all_game_files[100 % len(all_game_files)]  # Different game
        env2.assign_game_file(test_game_file2)
        
        obs3 = env2.reset()
        print(f"Different game - Game: {env2.assigned_game_file.split('/')[-3:]}")
        print(f"Different from first: {obs1 != obs3}")
        
        env2.close()
        
        # Test fallback behavior (no assigned game)
        print("\n=== Testing fallback behavior (no assigned game) ===")
        env3 = AlfredTXTEnv(config)
        obs4 = env3.reset(seed=42)
        print(f"Fallback mode works: {obs4 is not None}")
        env3.close()
    else:
        print("No game files found - ALFWORLD_DATA may not be set up")