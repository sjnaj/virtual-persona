#!/usr/bin/env python3
import logging
import yaml
from orchestrator import Orchestrator
from bot import VirtualPersonaBot

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ["httpx", "openai", "chromadb", "telegram", "httpcore"]:
        logging.getLogger(name).setLevel(logging.WARNING)

def main():
    setup_logging()
    config = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    
    orchestrator = Orchestrator(config)
    bot = VirtualPersonaBot(orchestrator, config["telegram"])
    
    logging.getLogger("main").info(
        f"启动 {config['persona']['name']}，"
        f"已有 {len(orchestrator.relationships.profiles)} 段关系"
    )
    bot.run()

if __name__ == "__main__":
    main()