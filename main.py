import asyncio

import hydra
from omegaconf import DictConfig

from dethy_rl.train import run_training


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    asyncio.run(run_training(cfg))


if __name__ == "__main__":
    main()
