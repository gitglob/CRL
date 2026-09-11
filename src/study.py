"""The actor-critic showcase is the default; explicit old configs retain the DQN study."""
import argparse

from .utils.config import load_config


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="config/showcase.yaml")
    args, _ = parser.parse_known_args()
    if load_config(args.config).get("experiment") == "showcase":
        from .showcase.study import main as run
    else:
        from .utils.study import main as run
    run()


if __name__ == "__main__":
    main()
