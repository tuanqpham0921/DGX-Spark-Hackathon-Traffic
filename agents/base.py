"""Base agent class for TRAFFIX agents."""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

# from langchain_openai import ChatOpenAI
from utils.config_loader import get_config


class BaseAgent(ABC):
    """Abstract base class for all TRAFFIX agents."""
    
    def __init__(
        self,
        name: str,
        role: str,
        llm: None,
        temperature: float = 0.7,
        model: str = "gpt-4o"
    ):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            role: Agent role/description
            llm: Optional pre-configured LLM
            temperature: LLM temperature
            model: LLM model name
        """
        self.name = name
        self.role = role
        self.logger = logging.getLogger(f"traffix.agents.{name.lower()}")
        
        # Initialize LLM
        # TODO (TQP): This is where we will set up the LLM later
        # need to pass it in theconstructor args
        if llm is None:
            config = get_config()
            llm_config = config.get("llm", {})
        else:
            self.llm = llm
            
        self.logger.info(f"Initialized {self.name} - {self.role}")
        
    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and return updated state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        pass
        
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        
        Returns:
            System prompt string
        """
        return f"You are {self.name}, a {self.role} for a traffic analysis system."
        
    def format_messages(self, system_prompt: str, user_message: str) -> list:
        """
        Format messages for LLM.
        
        Args:
            system_prompt: System prompt
            user_message: User message
            
        Returns:
            List of message dictionaries
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

