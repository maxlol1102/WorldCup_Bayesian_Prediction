from flask import Flask, request, jsonify, send_from_directory
import os
from model import get_team_rankings, get_win_probability, load_analysis_data  # import from model.py

app = Flask(__name__, static_folder="../frontend", static_url_path="")

# ======================
# ROUTES
# ======================

@app.route("/")
def index():
    """Serve the main page."""
    return send_from_directory("../frontend", "index.html")

@app.route("/analysis")
def analysis():
    """Serve the analysis page."""
    return send_from_directory("../frontend", "analysis.html")

@app.route("/rankings")
def rankings():
    """Return team Elo rankings as JSON."""
    return jsonify(get_team_rankings())

@app.route("/winprob")
def winprob():
    """Return win probability between two teams using Bayesian model."""
    team1 = request.args.get("team1")
    team2 = request.args.get("team2")
    if not team1 or not team2:
        return jsonify({"error": "You must provide team1 and team2"}), 400
    try:
        result = get_win_probability(team1, team2)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ======================
# NEW ANALYSIS ENDPOINT
# ======================

@app.route("/analysis_data")
def analysis_data():
    """
    Returns 100 pre-generated random matchups with Bayesian model probabilities.
    This dataset is saved in /data/analysis_matches.csv and only generated once.
    """
    try:
        df = load_analysis_data()
        return df.to_json(orient="records")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================
# STATIC FILE HANDLING
# ==========================
@app.route("/<path:path>")
def static_proxy(path):
    """Serve static files from the frontend folder"""
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    else:
        return jsonify({"error": f"File '{path}' not found"}), 404
    
    
    
# ======================
# MAIN
# ======================
if __name__ == "__main__":
    app.run(debug=True)
