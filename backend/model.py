import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
import random

# =======================
# Data Loading & Cleaning
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

players = pd.read_csv(os.path.join(DATA_DIR, "WorldCupPlayers.csv"), encoding="utf-8")
matches = pd.read_csv(os.path.join(DATA_DIR, "WorldCupMatches.csv"), encoding="utf-8")
world_cup = pd.read_csv(os.path.join(DATA_DIR, "WorldCups.csv"), encoding="utf-8")

matches.dropna(subset=['Year'], inplace=True)

# Fix bad encodings and inconsistent names
names = matches[matches['Home Team Name'].str.contains('rn">', na=False)]['Home Team Name'].value_counts()
wrong = list(names.index)
correct = [name.split('>')[1] for name in wrong]

old_name = ['Germany FR', 'Maracan� - Est�dio Jornalista M�rio Filho',
            'Estadio do Maracana', "C�te d'Ivoire"]
new_name = ['Germany', 'Maracan Stadium', 'Maracan Stadium', "Cote d'Ivoire"]

wrong = wrong + old_name
correct = correct + new_name

for i, wr in enumerate(wrong):
    matches = matches.replace(wr, correct[i])

# =======================
# Bayesian Inference Core
# =======================
class BayesianTeamInference:
    def __init__(self, prior_mean=1500, prior_std=300, beta=400):
        """
        beta controls how sensitive win probability is to rating differences.
        Increasing beta -> smoother curve (less extreme probabilities).
        """
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.beta = beta
        self.teams = {}
        self.graph = nx.DiGraph()

    def add_match_result(self, team1, team2, wins1, losses1):
        for team in [team1, team2]:
            if team not in self.teams:
                self.teams[team] = {'mean': self.prior_mean, 'std': self.prior_std}
                self.graph.add_node(team)

        total_games = wins1 + losses1
        if total_games > 0:
            win_rate = wins1 / total_games
            self.graph.add_edge(team1, team2, games=total_games, win_rate=win_rate)

    def _win_probability(self, skill_diff):
        """Smooth logistic transformation."""
        return 1 / (1 + np.exp(-skill_diff / self.beta))

    def get_win_probability(self, team1, team2):
        if team1 not in self.teams or team2 not in self.teams:
            raise ValueError("Both teams must be in the graph")

        skill_diff = self.teams[team1]['mean'] - self.teams[team2]['mean']
        prob = self._win_probability(skill_diff)

        # Clamp probabilities to avoid extremes like 0.99 or 0.01
        return float(np.clip(prob, 0.05, 0.95))

# Initialize model
inference_model = BayesianTeamInference()

# =======================
# Build Graph from Matches
# =======================
for _, row in matches.iterrows():
    home, away = row['Home Team Name'], row['Away Team Name']
    home_goals, away_goals = row['Home Team Goals'], row['Away Team Goals']

    if home_goals > away_goals:
        inference_model.add_match_result(home, away, wins1=1, losses1=0)
        inference_model.add_match_result(away, home, wins1=0, losses1=1)
    elif home_goals < away_goals:
        inference_model.add_match_result(home, away, wins1=0, losses1=1)
        inference_model.add_match_result(away, home, wins1=1, losses1=0)
    else:
        # Treat draw as half-win for both
        inference_model.add_match_result(home, away, wins1=0.5, losses1=0.5)
        inference_model.add_match_result(away, home, wins1=0.5, losses1=0.5)

# =======================
# Graph Data for GNN
# =======================
def build_graph_data(inference_model):
    node_features = []
    team_to_index = {}

    for idx, team in enumerate(inference_model.graph.nodes):
        team_to_index[team] = idx
        mean = inference_model.teams[team]['mean']
        std = inference_model.teams[team]['std']
        node_features.append([mean, std, mean / 1000])  # scaled feature for stability

    node_features = torch.tensor(node_features, dtype=torch.float)

    edge_index, edge_attr = [], []
    for u, v in inference_model.graph.edges:
        u_idx, v_idx = team_to_index[u], team_to_index[v]
        edge_index.append([u_idx, v_idx])

        games = inference_model.graph[u][v]['games']
        win_rate = inference_model.graph[u][v]['win_rate']
        edge_attr.append([games, win_rate])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr), team_to_index

data, team_to_index = build_graph_data(inference_model)

# =======================
# Bayesian GNN
# =======================
class BayesianGNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Train GNN
model = BayesianGNN(input_dim=3, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = torch.nn.MSELoss()

for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    target = data.x[:, 0].view(-1, 1)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()

# =======================
# Update Team Ratings
# =======================
model.eval()
with torch.no_grad():
    updated_skills = model(data).squeeze()
    updated_skills = (updated_skills - updated_skills.min()) / (updated_skills.max() - updated_skills.min())
    updated_skills = 1000 + updated_skills * 1000  # rescale roughly between 1000–2000

    for team, idx in team_to_index.items():
        inference_model.teams[team]['mean'] = updated_skills[idx].item()

# =======================
# API Functions
# =======================
def get_team_rankings():
    normalized_teams = {}
    for t, info in inference_model.teams.items():
        clean_name = t.strip().replace("—", "").replace("-", "").strip()
        if clean_name in normalized_teams:
            normalized_teams[clean_name] = max(normalized_teams[clean_name], info["mean"])
        else:
            normalized_teams[clean_name] = info["mean"]

    sorted_teams = sorted(normalized_teams.items(), key=lambda x: x[1], reverse=True)
    return [{"team": t, "mean": round(info, 2)} for t, info in sorted_teams]

def get_win_probability(team1, team2):
    prob = inference_model.get_win_probability(team1, team2)
    return {"team1": team1, "team2": team2, "probability": round(prob, 4)}



# =======================
# Analysis Dataset Loader
# =======================
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")
ANALYSIS_FILE = os.path.join(DATA_FOLDER, "analysis_matches.csv")

def load_analysis_data():
    """Simply load pre-generated analysis dataset."""
    if not os.path.exists(ANALYSIS_FILE):
        raise FileNotFoundError(f"Analysis dataset not found at {ANALYSIS_FILE}")
    return pd.read_csv(ANALYSIS_FILE)














