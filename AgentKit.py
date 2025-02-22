from pathlib import Path
from typing import Dict, List, Optional, Union, Any
# from yaml_parser import YAMLParser
import yaml
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.base import BaseTool
from langchain.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
import logging
from dotenv import load_dotenv
import importlib
from inspect import signature, Parameter
from typing import get_type_hints
from pydantic import create_model
from AgentKitSchema import AgentKitSchema

load_dotenv()

logger = logging.getLogger(__name__)

class AgentKit:
    """
    Main class for AgentKit that handles configuration loading and agent initialization.
    Provides a simple interface to create and manage LangChain agents.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize AgentKit with a configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = None        
        self.tools: List[BaseTool] = []
        self.agent_executor = None
        self.root_dir = None
        
        # Load and validate configuration
        self._load_config()
        self._initialize_project()
        self._initialize_tools()
        self._initialize_agent()

    def _load_config(self) -> None:
        """Load and validate the configuration file."""
        try:
            with open(self.config_path, "r") as file:
                data = yaml.safe_load(file)
            self.config = AgentKitSchema().load(data)
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def _initialize_project(self) -> None:
        """Initialize the project configuration."""
        self.project = None

        if "project" in self.config:
            self.project = self.config["project"]
        
        if "root_dir" in self.project:
            self.root_dir = Path(self.project["root_dir"])
        else:
            self.root_dir = self.config_path.parent

        if not self.root_dir.exists():
            logger.error(f"Root directory does not exist: {self.root_dir}")
            raise FileNotFoundError(f"Root directory does not exist: {self.root_dir}")

    def _initialize_tools(self) -> List[BaseTool]:
        """
        Initialize tools from the configuration.
        
        Returns:
            List[BaseTool]: List of initialized LangChain tools
        """
        initialized_tools = []

        if "tools" in self.config:
            tools_config = self.config["tools"]
            for tool_config in tools_config:
                try:
                    tool = self._load_tool(tool_config)
                    if tool:
                        initialized_tools.append(tool)
                except Exception as e:
                    logger.error(f"Failed to initialize tool {tool_config.get('id')}: {str(e)}")
        else:
            logger.warning("No tools found in configuration")
        
        return initialized_tools
    
    def _load_tool(self, tool_config: Dict) -> Optional[BaseTool]:
        """
        Load a single tool from its configuration.
        
        Args:
            tool_config: Tool configuration dictionary
            
        Returns:
            Optional[BaseTool]: Initialized tool or None if initialization fails
        """
        tool_path = Path(tool_config['path'])
        if not tool_path.is_absolute():
            if self.root_dir.is_absolute():
                tool_path = (self.root_dir / tool_path).resolve()
            else:
                tool_path = (self.config_path / Path("..") / self.root_dir / tool_path).resolve()
        
        if tool_path.exists():
            try:
                spec = importlib.util.spec_from_file_location(tool_path.stem, tool_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Get the function from the module
                func = getattr(module, tool_config.get('id'))
                
                # Get function signature and type hints
                sig = signature(func)
                type_hints = get_type_hints(func)
                
                # If the function has a single parameter, use regular Tool
                if len(sig.parameters) == 1:
                    param_name = next(iter(sig.parameters))
                    return Tool(
                        name=tool_config.get('id'),
                        description=tool_config.get('description'),
                        func=func
                    )
                
                # For multiple parameters, create a Pydantic model for the inputs
                fields = {}
                for name, param in sig.parameters.items():
                    param_type = type_hints.get(name, str)  # Default to str if no type hint
                    default = ... if param.default == Parameter.empty else param.default
                    fields[name] = (param_type, default)
                
                # Create a Pydantic model for the tool's arguments
                args_model = create_model(f"{tool_config.get('id')}Args", **fields)
                
                return StructuredTool.from_function(
                    func=func,
                    name=tool_config.get('id'),
                    description=tool_config.get('description'),
                    args_schema=args_model
                )
                
            except Exception as e:
                logger.error(f"Failed to load tool {tool_config.get('id')}: {str(e)}")
                return None
        else:
            logger.error(f"Tool file not found: {tool_path}")
            return None
        
    def _initialize_agent(self, 
                        agent_type: AgentType = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                        **kwargs) -> AgentExecutor:
        """
        Initialize a LangChain agent with the configured tools.
        
        Args:
            agent_type: Type of LangChain agent to initialize
            **kwargs: Additional arguments to pass to the agent initialization
            
        Returns:
            AgentExecutor: Initialized agent executor
        """
        # Initialize tools if not already done
        if not self.tools:
            self.tools = self._initialize_tools()
        
        # Get model configuration
        if "models" in self.config:
            models_config = self.config["models"]
            default_model = next((m for m in models_config if m.get('name') == 'gpt-4'), None)
        else:
            default_model = None
        
        # Initialize the language model
        llm = ChatOpenAI(
            temperature=default_model.get('temperature', 0.7) if default_model else 0.7,
            model_name=default_model.get('model_name', 'gpt-4') if default_model else 'gpt-4'
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are an AI assistant that helps users by using available tools to complete tasks.
                
                Remember:
                1. Use the tools provided to complete the task.
                2. No general conversation or greetings
                """),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        
        # Initialize the agent with handle_parsing_errors=True
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=agent_type,
            verbose=True,
            handle_parsing_errors=True,
            prompt=prompt,
            **kwargs
        )
        
        return self.agent_executor
    
    def run(self, input_text: str) -> str:
        """
        Run the agent with the given input.
        
        Args:
            input_text: The input text to process
            
        Returns:
            str: The agent's response
            
        Raises:
            ValueError: If the agent hasn't been initialized
        """
        if not self.agent_executor:
            raise ValueError("Agent not initialized. Call initialize_agent() first.")
            
        try:
            return self.agent_executor.invoke({"input": input_text})["output"]
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            raise