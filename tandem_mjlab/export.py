from pathlib import Path


def export_checkpoint(checkpoint: Path, output: Path) -> None:
    import torch
    from tensordict import TensorDict

    from tandem_mjlab.policy import WheelLegActor

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    current_dim = saved["actor_state_dict"]["mlp.0.weight"].shape[1] - 3
    history_dim = saved["actor_state_dict"]["encoder.0.weight"].shape[1]
    obs = TensorDict(
        {"actor": torch.zeros(1, current_dim), "history": torch.zeros(1, history_dim)},
        batch_size=[1],
    )
    config = {
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
    }
    actor = WheelLegActor(
        obs,
        {"actor": ["actor", "history"]},
        "actor",
        6,
        distribution_cfg=config,
    )
    actor.load_state_dict(saved["actor_state_dict"])
    module = actor.eval().as_onnx().eval()
    torch.onnx.export(
        module,
        module.get_dummy_inputs(),
        str(output),
        input_names=module.input_names,
        output_names=module.output_names,
        opset_version=18,
        dynamo=False,
    )


def main() -> None:
    import argparse

    from tandem_mjlab.source import PACKAGE_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=PACKAGE_DIR / "policies/locomotion.pt"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
