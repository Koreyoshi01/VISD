from __future__ import annotations


def validate_phase2_host_batch(inputs) -> None:
    batch_size = len(inputs)
    if batch_size != 1:
        raise ValueError(
            "VISD trainer currently expects a single-example host batch "
            f"(got {batch_size}). Keep `per_device_train_batch_size=1` for this host path."
        )

