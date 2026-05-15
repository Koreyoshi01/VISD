"""Compatibility shim for the canonical phase2 GRPO implementation."""


if __name__ == "__main__":
    from src.open_r1.phase2.grpo import GRPOConfig, GRPOScriptArguments, ModelConfig, TrlParser, main

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
else:
    from src.open_r1.phase2.grpo import *  # noqa: F401,F403
