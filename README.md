# Greedy agent

## Setup

```bash
conda create --name=greedy-agent python=3.10
conda activate greedy-agent
pip install -r requirements.txt
```

First, download and split the dataset, then run the agent:

```bash
python dataset_split.py
python run.py
```

The filtered action files `top_actions_1000.json` and `top_actions_2000.json`
contain the top 1,000 and 2,000 actions, ordered by frequency of use. They were
obtained by running ca. 1,000 chronics from the training split with
the same greedy agent logic but with all unitary actions (ca. 65,000).
