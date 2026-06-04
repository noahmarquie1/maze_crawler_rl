from stable_baselines3 import PPO
from agent import CrawlEnv
from kaggle_environments import make

MODEL_PATH = "ppo_crawl"
ACTION_MAPPING = {
    0: "BUILD_SCOUT",
    1: "BUILD_WORKER",
    2: "BUILD_MINER",
    3: "JUMP_NORTH",
    4: "JUMP_SOUTH",
    5: "JUMP_EAST",
    6: "JUMP_WEST",
}

# Agent function compatible with Kaggle Environment
crawl_env = CrawlEnv()
format_obs = lambda obs: crawl_env.format_obs(obs)


def my_agent(obs, config, agent: PPO):
    actions = {}
    for uid, data in obs.robots.items():
        rtype, col, row, energy, owner = data[0], data[1], data[2], data[3], data[4]
        if owner != obs.player:
            continue
        if rtype == 0:  # Factory
            # Perform RL model action (only controls Factory)
            agent_obs = crawl_env.format_obs(obs)
            action, _ = agent.predict(agent_obs)
            actions[uid] = ACTION_MAPPING[int(action)]

        else:
            actions[uid] = "NORTH"
    return actions


# Load model - Ensure it is written 
model = PPO.load(MODEL_PATH)
kaggle_agent = lambda obs, config: my_agent(obs, config, model)

# Test agent on single episode
kaggle_env = make("crawl", configuration={"randomSeed": 42})
kaggle_env.run([kaggle_agent, "random"])

# Optional - debugging mode (does not render)
"""

trainer = kaggle_env.train([None, "random"])
obs = trainer.reset()
done = False
while not done:
    action = my_agent(obs, None, model)
    obs, reward, done, info = trainer.step(action)

"""


html_out = kaggle_env.render(mode="html", width=800, height=800)
with open("replay.html", "w") as f:
    f.write(html_out)

print(f"Game finished successfully. Written to replay.html")

