from constants import ACTION_MAPPING, DEBUG, MODEL_PATH

from stable_baselines3 import PPO
from env import CrawlEnv
from kaggle_environments import make

# Agent function compatible with Kaggle Environment
crawl_env = CrawlEnv()


def my_agent(obs, config, agent: PPO, timestep=None):
    actions = {}
    for uid, data in obs.robots.items():
        rtype, col, row, energy, owner = data[0], data[1], data[2], data[3], data[4]
        if owner != obs.player:
            continue
        if rtype == 0:  # Factory
            # Perform RL model action (only controls Factory)
            agent_obs = crawl_env.format_obs(obs, crawl_env.timestep)
            action, _ = agent.predict(agent_obs)
            if DEBUG:
                print(ACTION_MAPPING[int(action)])
            actions[uid] = ACTION_MAPPING[int(action)]

        else:
            actions[uid] = "NORTH"
    return actions


if __name__ == "__main__":
    # Load model - Ensure it is written
    model = PPO.load(MODEL_PATH)
    kaggle_agent = lambda obs, config: my_agent(obs, config, model)
    kaggle_env = make("crawl", configuration={"randomSeed": 42})

    # Test agent on single episode
    if not DEBUG:
        kaggle_env.run([kaggle_agent, "random"])

    # Optional - debugging mode (does not render)
    else:
        trainer = kaggle_env.train([None, "random"])
        obs = trainer.reset()
        done = False
        while not done:
            action = my_agent(obs, None, model)
            obs, reward, done, info = trainer.step(action)

    html_out = kaggle_env.render(mode="html", width=800, height=800)
    with open("replay.html", "w") as f:
        f.write(html_out)

    print(f"Game finished successfully. Written to replay.html")
