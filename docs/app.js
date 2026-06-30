const data = window.FOOTBALL_PREDICTOR_DATA || {teams: [], venues: []};
const $ = selector => document.querySelector(selector);
const pct = value => `${(value * 100).toFixed(1)}%`;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

function setup() {
  const teams = data.teams.map(team => team.name).sort();
  $("#homeTeam").innerHTML = teams.map(team => `<option>${team}</option>`).join("");
  $("#awayTeam").innerHTML = teams.map(team => `<option>${team}</option>`).join("");
  $("#venue").innerHTML = [`<option value="">Neutral</option>`].concat(
    data.venues.map((venue, index) => `<option value="${index}">${venue.stadium_name} · ${venue.city}</option>`)
  ).join("");
  $("#homeTeam").value = teams.includes("Mexico") ? "Mexico" : teams[0];
  $("#awayTeam").value = teams.includes("France") ? "France" : teams[1];
  $("#simulateBtn").addEventListener("click", simulate);
  ["homeTeam", "awayTeam", "venue", "runs", "mode"].forEach(id => $(`#${id}`).addEventListener("change", simulate));
  $("#dataStatus").textContent = `${data.teams.length} equipos · ${data.venues.length} sedes`;
  simulate();
}

function team(name) {
  return data.teams.find(item => item.name === name) || data.teams[0];
}

function venueEdge(home, away) {
  const selected = $("#venue").value;
  if (selected === "") return 0.025;
  const venue = data.venues[Number(selected)];
  const city = `${venue.city || ""} ${venue.country || ""}`.toLowerCase();
  let edge = 0.012;
  if (city.includes(home.name.toLowerCase())) edge += 0.05;
  if (city.includes("mexico") && home.name === "Mexico") edge += 0.08;
  if (city.includes("mexico") && away.name === "Mexico") edge -= 0.08;
  if ((venue.altitude_m || 0) > 1200) edge += home.name === "Mexico" ? 0.025 : 0.005;
  return clamp(edge, -0.12, 0.12);
}

function baseRates(home, away) {
  const edge = venueEdge(home, away);
  const homePower = home.strength_score * .48 + home.attack_score * .34 + home.defense_score * .18;
  const awayPower = away.strength_score * .48 + away.attack_score * .34 + away.defense_score * .18;
  const diff = (homePower - awayPower) / 100 + edge;
  const homeGoals = clamp(1.25 + diff * 1.65 + (home.attack_score - away.defense_score) / 130, .25, 4.2);
  const awayGoals = clamp(1.08 - diff * 1.45 + (away.attack_score - home.defense_score) / 135, .2, 4.0);
  return {homeGoals, awayGoals, edge};
}

function poisson(lambda) {
  const limit = 7;
  const probs = [];
  let sum = 0;
  for (let goals = 0; goals <= limit; goals++) {
    const p = Math.exp(-lambda) * Math.pow(lambda, goals) / factorial(goals);
    probs.push(p);
    sum += p;
  }
  probs[limit] += Math.max(0, 1 - sum);
  return probs;
}

function factorial(n) {
  let result = 1;
  for (let i = 2; i <= n; i++) result *= i;
  return result;
}

function simulate() {
  const home = team($("#homeTeam").value);
  const away = team($("#awayTeam").value);
  if (!home || !away || home.name === away.name) return;

  const rates = baseRates(home, away);
  const hp = poisson(rates.homeGoals);
  const ap = poisson(rates.awayGoals);
  let homeWin = 0;
  let draw = 0;
  let awayWin = 0;
  let over = 0;
  let btts = 0;
  let topScore = "0-0";
  let topScoreProb = 0;

  for (let h = 0; h < hp.length; h++) {
    for (let a = 0; a < ap.length; a++) {
      const p = hp[h] * ap[a];
      if (h > a) homeWin += p;
      else if (h === a) draw += p;
      else awayWin += p;
      if (h + a > 2.5) over += p;
      if (h > 0 && a > 0) btts += p;
      if (p > topScoreProb) {
        topScoreProb = p;
        topScore = `${h}-${a}`;
      }
    }
  }

  render(home, away, {homeWin, draw, awayWin, over, btts, topScore, ...rates});
}

function render(home, away, result) {
  $("#homeName").textContent = home.name;
  $("#awayName").textContent = away.name;
  $("#homeSeed").textContent = fifaLabel(home);
  $("#awaySeed").textContent = fifaLabel(away);

  setProb("home", result.homeWin);
  setProb("draw", result.draw);
  setProb("away", result.awayWin);

  $("#scorePick").textContent = result.topScore;
  $("#overPick").textContent = pct(result.over);
  $("#bttsPick").textContent = pct(result.btts);
  $("#venueEdge").textContent = `${(result.edge * 100).toFixed(1)} pts`;
  $("#winnerHint").textContent = result.homeWin > result.awayWin ? home.name : away.name;

  const factors = [
    ["Fuerza", home.strength_score, away.strength_score],
    ["Ataque", home.attack_score, away.attack_score],
    ["Defensa", home.defense_score, away.defense_score],
    ["Ranking FIFA", rankScore(home), rankScore(away)],
    ["xG esperado", result.homeGoals * 40, result.awayGoals * 40],
  ];
  $("#factors").innerHTML = factors.map(([label, h, a]) => `
    <div class="factor">
      <span>${label}</span>
      <strong>${Number(h).toFixed(1)} / ${Number(a).toFixed(1)}</strong>
    </div>
  `).join("");

  const shotsHome = 7 + result.homeGoals * 4.2;
  const shotsAway = 7 + result.awayGoals * 4.2;
  $("#markets").innerHTML = [
    ["xG local", result.homeGoals.toFixed(2)],
    ["xG visitante", result.awayGoals.toFixed(2)],
    ["Tiros local", shotsHome.toFixed(1)],
    ["Tiros visitante", shotsAway.toFixed(1)],
    ["Corners total", (6.2 + (result.homeGoals + result.awayGoals) * 1.3).toFixed(1)],
    ["Tarjetas total", (3.1 + Math.abs(home.strength_score - away.strength_score) / 35).toFixed(1)],
  ].map(([label, value]) => `<div class="market"><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function setProb(kind, value) {
  $(`#${kind}Prob`).textContent = pct(value);
  $(`#${kind}Bar`).style.width = pct(value);
}

function fifaLabel(teamData) {
  return teamData.fifa_rank ? `FIFA #${teamData.fifa_rank}` : "Equipo";
}

function rankScore(teamData) {
  return clamp(100 - ((teamData.fifa_rank || 90) - 1) * 1.2, 20, 100);
}

setup();
