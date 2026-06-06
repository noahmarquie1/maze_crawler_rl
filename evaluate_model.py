from agent import CrawlEnv
from main import MODEL_PATH
from stable_baselines3 import PPO
from agent import CrawlEnv
from kaggle_environments import make

crawl_env = CrawlEnv()

model = PPO.load(MODEL_PATH)
obs, _ = crawl_env.reset()

done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = crawl_env.step(action.item())

html_out = crawl_env.render(mode="html", width=800, height=800)
with open("replay.html", "w") as f:
    f.write(html_out)
