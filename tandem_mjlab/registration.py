from mjlab.tasks.registry import register_mjlab_task

from tandem_mjlab.function.locomotion.task import env_cfg, ppo_cfg
from tandem_mjlab.runner import WheelLegRunner

register_mjlab_task(
    task_id="Mjlab-Tandem-WheelLeg-Flat",
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(),
    rl_cfg=ppo_cfg(),
    runner_cls=WheelLegRunner,
)
