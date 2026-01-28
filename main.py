"""Main entry point for Walkie AI agent.

This module provides the main execution pipeline for the agent.
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agent.core import WalkieAgent, create_agent_from_config
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()

# Setup logger
logger = setup_logger(__name__)


def create_agent(verbose: bool = False) -> WalkieAgent:
    """Create and configure the Walkie agent.
    
    Args:
        verbose: Enable verbose logging.
        
    Returns:
        Configured WalkieAgent instance.
    """
    agent = create_agent_from_config(verbose=verbose)
    logger.info("Walkie agent initialized successfully")
    return agent


async def process_text_command(agent: WalkieAgent, command: str) -> str:
    """Process a text command through the agent.
    
    Args:
        agent: WalkieAgent instance.
        command: Text command to process.
        
    Returns:
        Agent response.
    """
    logger.info(f"Processing command: {command}")
    response = await agent.execute_task(command)
    logger.info(f"Agent response: {response}")
    return response


async def process_voice_command(agent: WalkieAgent, audio_path: Path) -> tuple[str, bytes]:
    """Process a voice command through the agent.
    
    Args:
        agent: WalkieAgent instance.
        audio_path: Path to audio file.
        
    Returns:
        Tuple of (transcribed_text, audio_response).
    """
    # Read audio file
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    logger.info(f"Processing voice input from: {audio_path}")
    
    # Process voice input
    text, response = await agent.process_voice_input(audio_bytes)
    logger.info(f"Transcribed: {text}")
    logger.info(f"Response: {response}")
    
    # Generate voice response
    audio_response = await agent.generate_voice_response(response)
    
    return response, audio_response


async def interactive_mode(agent: WalkieAgent) -> None:
    """Run agent in interactive text mode.
    
    Args:
        agent: WalkieAgent instance.
    """
    print("Walkie AI Agent - Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = await agent.execute_task(user_input)
            print(f"\nAgent: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"\nError: {e}")


async def main() -> None:
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Walkie AI Agent")
    parser.add_argument(
        "--mode",
        choices=["interactive", "text", "voice"],
        default="interactive",
        help="Execution mode",
    )
    parser.add_argument(
        "--command",
        type=str,
        help="Text command to execute (for text mode)",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Audio file path (for voice mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output audio file path (for voice mode)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Create agent
    agent = create_agent(verbose=args.verbose)
    
    try:
        if args.mode == "interactive":
            await interactive_mode(agent)
        
        elif args.mode == "text":
            if not args.command:
                print("Error: --command required for text mode")
                return
            
            response = await process_text_command(agent, args.command)
            print(f"\nResponse: {response}")
        
        elif args.mode == "voice":
            if not args.audio:
                print("Error: --audio required for voice mode")
                return
            
            response, audio_bytes = await process_voice_command(agent, args.audio)
            print(f"\nResponse: {response}")
            
            if args.output:
                with open(args.output, "wb") as f:
                    f.write(audio_bytes)
                print(f"Audio saved to: {args.output}")
    
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
