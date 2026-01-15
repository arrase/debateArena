import argparse
import sys
import os
from debate_arena.utils.config_loader import load_config
from debate_arena.core.manager import DebateManager

def main():
    parser = argparse.ArgumentParser(description="Autonomous Debate CLI System")
    parser.add_argument('-p', '--prompt', type=str, help="The debate topic", required=False)
    parser.add_argument('--config', type=str, default="config/settings.yaml", help="Path to configuration file")
    parser.add_argument('-f', '--file', type=str, help="Output file to save the debate transcript", required=False)
    
    args = parser.parse_args()
    
    try:
        # Load Config
        # If running as installed package, config might need simpler resolution or user passed path
        config_path = args.config
        if not os.path.exists(config_path):
             # Try determining relative to current working dir if not found
             pass
             
        config = load_config(config_path)
        
        # Override topic if provided via CLI
        topic = args.prompt if args.prompt else config['debate']['topic']
        
        if not topic:
            print("Error: No topic provided via CLI (-p) or config file.")
            sys.exit(1)
            
        # Initialize and Run Manager
        manager = DebateManager(config, topic, output_file=args.file)
        manager.run_debate()
        
    except Exception as e:
        print(f"Fatal Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
