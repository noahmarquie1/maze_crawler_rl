# Maze Crawler RL - Kaggle

Competition entry for Kaggle Maze Crawler by Noah Marquié and Michael Dickinson. Using PPO RL architecture to learn competitive maze traversal.

Final model trained using 24 CPU cores and one Nvidia H100 GPU for 3 hours on SFU's Fir Supercomputer.

## Preview

<img src="preview.gif" width=60% alt="Maze Crawler Robot Preview" />


## Usage

Install repo and configure environment - Python 3.11.0 Recommended
```
git clone https://github.com/noahmarquie1/maze_crawler_rl.git
pip install -e .
```

### Train Agent
```
python train.py
```

### Run model

Outputs to `replay.html` by default, which can be viewed in-browser

```
python main.py
```

### Acknowledgements

Computational resources provided by the Digital Research Alliance of Canada (Fir Cluster, Simon Fraser University).

Bovard Doerschuk-Tiberi. Maze Crawler. https://kaggle.com/competitions/maze-crawler, 2026. Kaggle.
