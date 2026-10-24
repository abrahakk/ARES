from __future__ import annotations

import argparse
import logging

import numpy as np

from rl.environments.observation_space import OBSERVATION_DIM

logger = logging.getLogger(__name__)

VALIDATION_BATCH_SIZE = 32
VALIDATION_TOLERANCE = 1e-4


class _ExportWrapper:

    def __new__(cls, actor):
        import torch.nn as nn

        class _Wrapped(nn.Module):
            def __init__(self, inner_actor):
                super().__init__()
                self.inner_actor = inner_actor

            def forward(self, observation):
                return self.inner_actor.export_forward(observation)

        return _Wrapped(actor)


def export_and_validate(checkpoint_path: str, output_path: str, device: str = "cpu") -> None:
    import torch

    from rl.networks.actor import ActorNetwork

    actor = ActorNetwork(OBSERVATION_DIM).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["actor"] if "actor" in checkpoint else checkpoint)
    actor.eval()

    dummy_input = torch.randn(1, OBSERVATION_DIM, device=device)
    export_module = _ExportWrapper(actor)
    export_module.eval()

    logger.info("exporting to %s", output_path)
    torch.onnx.export(
        export_module,
        dummy_input,
        output_path,
        input_names=["observation"],
        output_names=["action", "value"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
    )
    logger.info("export complete")

    _validate_export(actor, output_path, device)


def _validate_export(actor, onnx_path: str, device: str) -> None:

    import onnxruntime as ort
    import torch

    rng = np.random.default_rng(42)
    test_batch = rng.standard_normal((VALIDATION_BATCH_SIZE, OBSERVATION_DIM)).astype(np.float32)

    with torch.no_grad():
        torch_action, torch_value = actor.export_forward(torch.as_tensor(test_batch, device=device))
    torch_action = torch_action.cpu().numpy()
    torch_value = torch_value.cpu().numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_action, onnx_value = session.run(None, {input_name: test_batch})

    action_diff = np.max(np.abs(torch_action - onnx_action))
    value_diff = np.max(np.abs(torch_value - onnx_value))

    logger.info("validation: max action diff=%.6f, max value diff=%.6f (tolerance=%.6f)",
                action_diff, value_diff, VALIDATION_TOLERANCE)

    if action_diff > VALIDATION_TOLERANCE or value_diff > VALIDATION_TOLERANCE:
        raise RuntimeError(
            f"ONNX export validation FAILED: action_diff={action_diff:.6f}, "
            f"value_diff={value_diff:.6f} exceed tolerance={VALIDATION_TOLERANCE}. "
            "Do not deploy this export - the ONNX model does not match the trained PyTorch model."
        )
    logger.info("validation PASSED - ONNX export matches the PyTorch model within tolerance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/onnx/ares_policy.onnx")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    export_and_validate(args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    main()

