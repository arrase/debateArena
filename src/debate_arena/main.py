import argparse
import sys
from pathlib import Path

from debate_arena.core.manager import DebateManager
from debate_arena.utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description="Autonomous Debate CLI System")
    parser.add_argument("-p", "--prompt", help="The debate topic")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to configuration file")
    parser.add_argument("-f", "--file", help="Output file to save the debate transcript")

    args = parser.parse_args()

    try:
        config_path = Path(args.config)
        if not config_path.exists():
            root_path = Path(__file__).resolve().parents[2]
            config_path = root_path / args.config

        config = load_config(str(config_path))
        topic = args.prompt or config["debate"]["topic"]
        if not topic:
            print("Error: No topic provided via CLI (-p) or config file.")
            sys.exit(1)

        DebateManager(config, topic, output_file=args.file).run_debate()
    except Exception as e:
        print(f"Fatal Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
