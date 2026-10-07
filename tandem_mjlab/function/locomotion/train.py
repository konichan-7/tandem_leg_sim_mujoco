from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import time

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from tandem_mjlab.function.locomotion.task import env_cfg, ppo_cfg
from tandem_mjlab.model import MODEL_PATH
from tandem_mjlab.runner import WheelLegRunner
from tandem_mjlab.source import PACKAGE_DIR, source_files


def main() -> None:
    parser = argparse.ArgumentParser(description="从发布 checkpoint 续训轮速修正网络")
    parser.add_argument(
        "--checkpoint", type=Path, default=PACKAGE_DIR / "policies/locomotion.pt"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--steering-learning-rate", type=float, default=5e-4)
    parser.add_argument("--mini-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = env_cfg(randomize=args.randomize)
    cfg.scene.num_envs = args.num_envs
    agent = ppo_cfg()
    agent.seed = args.seed
    agent.algorithm.learning_rate = args.learning_rate
    agent.algorithm.num_mini_batches = args.mini_batches
    agent.save_interval = max(1, args.iterations // 3)
    agent.upload_model = False
    agent.run_name = args.output.name
    agent.max_iterations = args.iterations
    state = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "sample_budget": args.num_envs * agent.num_steps_per_env * args.iterations,
        "randomize": args.randomize,
        "learning_rate": args.learning_rate,
        "steering_learning_rate": args.steering_learning_rate,
        "mini_batches": args.mini_batches,
        "optimizer_restored": False,
    }

    def update(phase: str, **fields) -> None:
        state.update(
            phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields
        )
        temporary = args.output / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.output / "status.json")
        print(json.dumps(state), flush=True)

    update("preparing")
    try:
        with tarfile.open(args.output / "source_snapshot.tar.gz", "w:gz") as archive:
            for path in source_files():
                archive.add(
                    path, arcname=f"tandem_mjlab/{path.relative_to(PACKAGE_DIR)}"
                )
            archive.add(PACKAGE_DIR.parent / "MJCF/demo", arcname="MJCF/demo")
        (args.output / "agent.json").write_text(
            json.dumps(asdict(agent), indent=2) + "\n"
        )
        env = ManagerBasedRlEnv(cfg, device=args.device)
        try:
            runner = WheelLegRunner(
                RslRlVecEnvWrapper(env),
                asdict(agent),
                str(args.output),
                device=args.device,
            )
            runner.alg.load(
                saved,
                {"actor": True, "critic": True, "optimizer": False, "iteration": False},
                True,
            )
            runner.alg.optimizer = torch.optim.Adam(
                [
                    {
                        "params": runner.alg.actor.steering.parameters(),
                        "lr": args.steering_learning_rate,
                    },
                    {
                        "params": runner.alg.critic.parameters(),
                        "lr": args.learning_rate,
                    },
                ]
            )
            with torch.no_grad():
                runner.alg.actor.distribution.std_param[:4].fill_(0.005)
                runner.alg.actor.distribution.std_param[4:].fill_(0.01)
            frozen = {
                name: value.clone()
                for name, value in runner.alg.actor.state_dict().items()
                if not name.startswith("steering.")
            }
            runner.save(str(args.output / "warm_start.pt"))
            update("training", stand_leg_alpha=runner.alg.actor.stand_leg_alpha.item())
            started = time.monotonic()
            runner.learn(
                num_learning_iterations=args.iterations, init_at_random_ep_len=True
            )
            elapsed = time.monotonic() - started
            if not all(
                torch.equal(value, runner.alg.actor.state_dict()[name])
                for name, value in frozen.items()
            ):
                raise RuntimeError("Frozen base policy weights changed")
            if not all(
                torch.isfinite(value).all()
                for value in runner.alg.actor.state_dict().values()
            ):
                raise RuntimeError("Nonfinite trained actor")
            update(
                "training_complete",
                elapsed_s=elapsed,
                samples_per_s=state["sample_budget"] / elapsed,
                final_checkpoint=str(
                    args.output / f"model_{runner.current_learning_iteration}.pt"
                ),
                base_weights_bitwise_preserved=True,
            )
        finally:
            env.close()
    except Exception as error:
        update("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
