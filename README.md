# Game Theory and Optimisation Assignment

This project implements a shortest path algorithm for the egg transport optimization problem using Dijkstra's algorithm on a city-like road network.

## Setup Instructions

### 1. Virtual Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (Git Bash):
source venv/Scripts/activate
# On Windows (Command Prompt):
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Quick Setup
Alternatively, you can use the provided activation script:
```bash
source activate_env.sh
```

### 3. Running the Program
```bash
python "Game Theory and Optimisation/main.py"
```

## Project Structure
- `Game Theory and Optimisation/main.py` - Main program implementing Dijkstra's algorithm
- `Game Theory and Optimisation/graph_gen.py` - City grid graph generator
- `Game Theory and Optimisation/viz.py` - Visualization functions for graphs
- `Game Theory and Optimisation/network_handler.py` - Handles different network sizes and output formats
- `Game Theory and Optimisation/analytics.py` - Algorithm comparison and analytics module
- `output/` - Directory containing all generated files (created automatically)
  - `shortest_path_[N]nodes_[timestamp].txt` - Path files for networks > 50 nodes
  - `breakage_matrix_[N]nodes_[timestamp].npz` - CSR matrix files for networks > 200 nodes
  - `algorithm_comparison_[N]nodes_[timestamp].txt` - Analytics comparison reports
- `requirements.txt` - Python package dependencies
- `venv/` - Virtual environment (created after setup)

## Description
The main file contains all we need to solve the problem. Each function is documented explaining what it does. The program generates a random city-like grid graph representing a road network where each edge weight represents the breakage of eggs on that road. It then calculates the path from starting node s to target node t with minimum total breakage using Dijkstra's algorithm.
