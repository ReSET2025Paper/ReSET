'''
Dataset Structure:
    Expects a directory of pickle files (e.g., demo_0.pkl, demo_1.pkl, ...),
    All lists must have the same length T (number of timesteps per episode).
    each containing a dictionary with the following keys:

    - 'img'         : list of T BGR frames (np.ndarray, shape [H, W, 3], uint8)
                      → mapped to LeRobot feature "image"
    - 'img_gripper' : list of T BGR frames (np.ndarray, shape [H, W, 3], uint8)
                      → mapped to LeRobot feature "wrist_image"
    - 'joint_state' : list of T joint states (np.ndarray, shape [8,], float32)
                      → mapped to LeRobot feature "state"
    - 'joint_vel'   : list of T joint velocity actions (np.ndarray, shape [8,], float32)
                      → mapped to LeRobot feature "actions"
    - 'text_cond'   : (optional) str, task language description
                      → if absent, falls back to cfg.description from preprocess_lerobot.yaml
                      → mapped to LeRobot feature "task" / "prompt"

    Preprocessing via preprocess/preprocess_lerobot.py:
        uv run preprocess/preprocess_lerobot.py task_name=[task_name] num_demos=[N] data_path=./robot_video/[task_name]/[type]
    Converts the above pickle files into a LeRobotDataset stored at:
        ~/.cache/huggingface/lerobot/[task_name]_[num_demos]/
'''

# -----------------------------------------------------------------------------
# Add the following to openpi/src/openpi/training/config.py:
@dataclasses.dataclass(frozen=True)
class ReSETDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group( # NOTE: We used the same configuration as in the original libero_policy.py, but you can modify it to match your dataset.
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

# -----------------------------------------------------------------------------
# Add this under _CONFIGS:
    TrainConfig(
        name="ReSET",
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=ReSETDataConfig(
            repo_id="[task_name_20]", # NOTE: change this to your own dataset name
            base_config=DataConfig(prompt_from_task=True),
        ),
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        batch_size=16,
    ),