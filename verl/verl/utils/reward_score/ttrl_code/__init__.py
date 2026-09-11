"""Code reward adapter for VERL custom reward loading.

The Eurus code data stores executable test cases in ``ground_truth``.  Route all
samples through VERL's code scorer while preserving the custom reward function
signature expected by the PPO trainer.
"""

from verl.utils.reward_score import default_compute_score


def reward_func(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Execute the generated program against the sample's test cases."""
    del data_source
    return default_compute_score(
        data_source="apps",
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        sandbox_fusion_url=sandbox_fusion_url,
        concurrent_semaphore=concurrent_semaphore,
        memory_limit_mb=memory_limit_mb,
        **kwargs,
    )
