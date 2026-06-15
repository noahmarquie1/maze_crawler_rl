"""Kaggle Crawl submission entrypoint.

Loads the trained MaskablePPO agent straight from its saved ``.zip`` archive and
serves it through the ``agent(obs, config)`` interface the competition calls.
Importing ``model`` first registers the custom ``CrawlMaskablePolicy`` so the
archive's pickled policy class resolves on load. Observation encoding and action
masking are reused verbatim from ``env`` -- there is no second copy of the
network here.
"""
import contextlib
import importlib.machinery
import os
import sys
import types

# In the competition the tarball is unpacked here (this file staged as main.py);
# running this file locally resolves assets next to it instead.
_AGENT_DIR = "/kaggle_simulations/agent"
BASE_DIR = (
    _AGENT_DIR if os.path.isdir(_AGENT_DIR) else os.path.dirname(os.path.abspath(__file__))
)

# sb3_contrib is not installed in the Kaggle runtime, and its stable_baselines3
# is a different build than the one this model was saved with. We vendor both at
# their training versions under _vendor/ and put that dir first on sys.path so
# the saved archive loads against matching code. (Empty/absent locally, where the
# venv already has them.)
sys.path.insert(0, os.path.join(BASE_DIR, "_vendor"))


class _StubAttr:
    """Stand-in for any symbol pulled from a stubbed module. Safe to use as a
    type annotation, base class, or callable -- none of which actually run at
    inference."""

    def __init__(self, *args, **kwargs):
        pass


class _StubModule(types.ModuleType):
    """Module whose every attribute resolves to ``_StubAttr``, so import-time
    references like the ``matplotlib.figure.Figure`` annotation in SB3's logger
    succeed without importing the real, broken package."""

    def __getattr__(self, name):
        # Let dunders (__file__, __path__, __spec__, ...) fall through to a
        # normal AttributeError so importlib/inspect can probe the module
        # safely; only shadow real symbols SB3 reads at import time.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _StubAttr


def _install_stub(name):
    """Shadow ``name`` with a lenient stub module before the SB3 import chain.

    The Kaggle competition image ships tensorboard and matplotlib builds that
    crash on import (broken protobuf/tensorflow chain; matplotlib's C extension
    compiled against numpy 1.x while the runtime has numpy 2.x). SB3 imports both
    eagerly -- for logging we never do at inference -- and the failures are not
    plain ImportErrors, so SB3's own guards don't catch them. Shadowing the
    modules sidesteps the whole problem.
    """
    module = _StubModule(name)
    # A real spec (with origin=None) keeps importlib.util.find_spec happy --
    # torch._dynamo enumerates some modules (e.g. pandas) that way and a None
    # __spec__ would raise ValueError.
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = module
    return module


# Must run before `import model` (which triggers the stable_baselines3 import).
_install_stub("torch.utils.tensorboard")
_install_stub("pandas")
_mpl = _install_stub("matplotlib")
_mpl.pyplot = _install_stub("matplotlib.pyplot")
_mpl.figure = _install_stub("matplotlib.figure")
# SB3's vec_video_recorder imports gymnasium.wrappers.monitoring, which the
# Kaggle runtime's newer gymnasium no longer ships. Video recording is unused at
# inference, so stub the missing submodule.
_gym_mon = _install_stub("gymnasium.wrappers.monitoring")
_gym_mon.video_recorder = _install_stub("gymnasium.wrappers.monitoring.video_recorder")

import model  # noqa: E402,F401  -- registers CrawlMaskablePolicy for unpickling
from env import compute_action_masks, format_obs, game_agent  # noqa: E402

MODEL_FILE = os.path.join(BASE_DIR, "ppo_crawl.zip")

# sb3 and kaggle_environments are noisy on import/load; suppress only during
# setup so the chatter can't corrupt the engine's stdout protocol. CPU keeps the
# agent portable across the competition's hardware (inference is ~20ms/step).
with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(
    open(os.devnull, "w")
):
    from sb3_contrib import MaskablePPO

    _model = MaskablePPO.load(MODEL_FILE, device="cpu")

# The agent has no CrawlEnv to count steps, so mirror its timestep: start at 0
# and advance once per call, matching how the stats channel is normalized in
# training (CrawlEnv formats step k's observation with timestep == k).
_timestep = 0


def agent(obs, config):
    global _timestep
    observation = format_obs(obs, _timestep)
    masks = compute_action_masks(obs)
    action, _ = _model.predict(observation, action_masks=masks, deterministic=True)
    _timestep += 1
    return game_agent(obs, action)


if __name__ == "__main__":
    # Local smoke test: play one game against the decision-tree opponent and
    # write a replay. Not used by the competition.
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        from kaggle_environments import make

        from opponent import decision_tree_opponent

        env = make("crawl")
        env.run([agent, decision_tree_opponent])

    with open("replay.html", "w") as f:
        f.write(env.render(mode="html", width=800, height=800))
    print("Game finished successfully. Wrote replay.html")
