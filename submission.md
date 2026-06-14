Find the Competition

kaggle competitions list -s "maze-crawler"
kaggle competitions pages maze-crawler
kaggle competitions pages maze-crawler --content

Accept the Competition Rules

Before submitting, you must accept the rules on the Kaggle website. Navigate to https://www.kaggle.com/competitions/maze-crawler and click "Join Competition".

Verify you've joined:

kaggle competitions list --group entered

Download Competition Data

kaggle competitions download maze-crawler -p crawl-data

Submit Your Agent

Your submission must have a main.py at the root with an agent function.

Single file agent:

kaggle competitions submit maze-crawler -f main.py -m "Worker rush v1"

Multi-file agent — bundle into a tar.gz with main.py at the root:

tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit maze-crawler -f submission.tar.gz -m "Multi-file agent v1"

Notebook submission:

kaggle competitions submit maze-crawler -k YOUR_USERNAME/crawl-agent -f submission.tar.gz -v 1 -m "Notebook agent v1"

Monitor Your Submission

Check submission status:

kaggle competitions submissions maze-crawler

Note the submission ID from the output — you'll need it for episodes.
List Episodes

Once your submission has played some games:

kaggle competitions episodes <SUBMISSION_ID>

CSV output for scripting:

kaggle competitions episodes <SUBMISSION_ID> -v

Download Replays and Logs

Download the replay JSON for an episode (for visualization or analysis):

kaggle competitions replay <EPISODE_ID>
kaggle competitions replay <EPISODE_ID> -p ./replays

Download agent logs to debug your agent's behavior:

# Logs for the first agent (index 0)
kaggle competitions logs <EPISODE_ID> 0

# Logs for the second agent (index 1)
kaggle competitions logs <EPISODE_ID> 1 -p ./logs

Check the Leaderboard

kaggle competitions leaderboard maze-crawler -s

Typical Workflow

# Test locally
python -c "
from kaggle_environments import make
env = make('crawl', debug=True)
env.run(['main.py', 'random'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"

# Submit
kaggle competitions submit maze-crawler -f main.py -m "v1"

# Check status
kaggle competitions submissions maze-crawler

# Review episodes
kaggle competitions episodes <SUBMISSION_ID>

# Download replay and logs
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0

# Check leaderboard
kaggle competitions leaderboard maze-crawler -s