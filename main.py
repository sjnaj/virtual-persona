#!/usr/bin/env python3
import logging
import yaml
from orchestrator import Orchestrator
from bot import VirtualPersonaBot

def setup_logging(log_dir: str = "log"):
    import os
    os.makedirs(log_dir, exist_ok=True)

    file_fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console — INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)

    # File: full log — INFO and above
    info_handler = logging.FileHandler(
        os.path.join(log_dir, "info.log"), mode="a", encoding="utf-8"
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(file_fmt)

    # File: warnings/errors only — WARNING and above
    warn_handler = logging.FileHandler(
        os.path.join(log_dir, "warn.log"), mode="a", encoding="utf-8"
    )
    warn_handler.setLevel(logging.WARNING)
    warn_handler.setFormatter(file_fmt)

    root.addHandler(console_handler)
    root.addHandler(info_handler)
    root.addHandler(warn_handler)

    # Suppress noisy third-party libraries
    for name in ["httpx", "openai", "telegram", "httpcore", "chromadb"]:
        logging.getLogger(name).setLevel(logging.WARNING)

def main():
    setup_logging()
    with open("config.local.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    orchestrator = Orchestrator(config)
    bot = VirtualPersonaBot(orchestrator, config["telegram"])
    
    logging.getLogger("main").info(
        f"启动 {config['persona']['name']}，"
        f"已有 {len(orchestrator.relationships.profiles)} 段关系"
    )
    bot.run()

if __name__ == "__main__":
    main()