"""Runtime shims for kaggle_environments to speed up training rollouts.

These poke library internals deliberately; keep them isolated here so the intent
is explicit and they are easy to remove if kaggle_environments changes.
"""
from copy import deepcopy

import kaggle_environments.core as kcore
from kaggle_environments.utils import default_schema

_SCHEMA_PATCHED = False


def patch_kaggle_schema_validation() -> None:
    """Skip the per-step jsonschema validation kaggle_environments runs.

    On every ``make`` and ``step`` the engine validates the (engine-produced)
    state and our already-mapped actions against a JSON schema. During training
    the action format is guaranteed by ``game_agent`` and the observation comes
    from the engine itself, so the validation is dead weight -- profiling showed
    it as the majority of env-step time. We keep the default-filling pass (needed
    for correctness) and drop only the ``jsonschema.validate`` call. Idempotent.
    """
    global _SCHEMA_PATCHED
    if _SCHEMA_PATCHED:
        return

    def fast_process_schema(schema, data, use_default=True):
        if use_default:
            data = default_schema(schema, deepcopy(data))
        return None, data

    kcore.process_schema = fast_process_schema
    _SCHEMA_PATCHED = True
