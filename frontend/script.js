// ===== Splash Screen Logic =====
window.addEventListener("load", () => {
  const splash = document.getElementById("splash");
  const main = document.getElementById("main-content");

  setTimeout(() => {
    splash.classList.add("fade-out");
    setTimeout(() => {
      splash.style.display = "none";
      main.style.display = "block";
      setTimeout(() => {
        main.classList.add("visible");
      }, 40);
    }, 950);
  }, 2000);
});

// ===== Load Teams =====
async function loadTeams() {
  const res = await fetch("/rankings");
  const rankings = await res.json();
  const teamSelect1 = document.getElementById("team1");
  const teamSelect2 = document.getElementById("team2");

  // Clear old options
  teamSelect1.innerHTML = "";
  teamSelect2.innerHTML = "";

  const seen = new Set();
  rankings.forEach(item => {
    if (!seen.has(item.team)) {
      const opt1 = document.createElement("option");
      opt1.value = item.team;
      opt1.textContent = item.team;

      const opt2 = document.createElement("option");
      opt2.value = item.team;
      opt2.textContent = item.team;

      teamSelect1.appendChild(opt1);
      teamSelect2.appendChild(opt2);
      seen.add(item.team);
    }
  });
}

// ===== Load Rankings =====
async function loadRankings() {
  try {
    const response = await fetch("/rankings");
    if (!response.ok) throw new Error("Failed to fetch rankings");

    const data = await response.json();
    const list = document.getElementById("rankings");
    list.innerHTML = "";

    data.forEach(item => {
      const li = document.createElement("li");
      li.textContent = `${item.team}: ${item.mean.toFixed(2)}`;
      list.appendChild(li);
    });
  } catch (err) {
    console.error(err);
  }
}

// ===== Compare Teams =====
async function calculateWinProb() {
  const team1 = document.getElementById("team1").value;
  const team2 = document.getElementById("team2").value;

  if (!team1 || !team2) {
    alert("Please select both teams!");
    return;
  }

  if (team1 === team2) {
    alert("Please select two different teams!");
    return;
  }

  // Clear results
  document.getElementById("result").innerText = "Calculating your model prediction...";
  ["chatgpt-result", "deepseek-result"].forEach(id => {
    document.getElementById(id).innerHTML = `<span class="loading"> Generating...</span>`;
  });

  try {
    // ===== Your Bayesian Model =====
    const res = await fetch(`/winprob?team1=${team1}&team2=${team2}`);
    if (!res.ok) throw new Error("Failed to fetch win probability");

    const data = await res.json();
    const prob1 = (data.probability * 100).toFixed(2);
    const prob2 = (100 - prob1).toFixed(2);

    document.getElementById("result").innerText =
      `${team1} vs ${team2}\n` +
      `${team1}: ${prob1}%\n` +
      `${team2}: ${prob2}%`;

    // ===== Question Template =====
    const prompt = `Who is more likely to win between ${team1} and ${team2} in football? Estimate the win probability for each team (X.XX% format, no draw, single match, neutral field). No explanations. Just provide the name and the percentages.`;

    // ===== AI Models =====
    const models = [
      { id: "chatgpt-result", name: "gpt-5-nano" },
      { id: "deepseek-result", name: "deepseek-chat" }
    ];

    for (const model of models) {
      puter.ai.chat(prompt, { model: model.name })
        .then(resp => {
          document.getElementById(model.id).innerText = resp;
        })
        .catch(err => {
          console.error(`Error for ${model.name}:`, err);
          document.getElementById(model.id).innerText = "⚠️ Error generating response.";
        });
    }
  } catch (err) {
    console.error(err);
    document.getElementById("result").innerText = "Error calculating probability.";
  }
}

// ===== On Page Load =====
window.onload = function () {
  loadTeams();
  loadRankings();
};
