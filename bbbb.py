# -*- coding: utf-8 -*-
"""
Universal Competition Tracker — Swiss, League, KO (individual & teams)
UI: Gradio | Exports: CSV & PDF (ReportLab)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple, Dict, Callable, Any, Union
import os, json, csv, math, random, itertools
from collections import defaultdict

import gradio as gr
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# ---------------------------
# Config & constantes
# ---------------------------
OUTPUT_DIR = "./torneo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
STATE_FILE = os.path.join(OUTPUT_DIR, "tournament_state.json")

# Sistemas de puntuación predefinidos (ampliables)
SCORING_PRESETS: Dict[str, Dict[str, float]] = {
    "Ajedrez": {"win": 1.0, "draw": 0.5, "loss": 0.0, "bye": 1.0},
    "Fútbol": {"win": 3.0, "draw": 1.0, "loss": 0.0, "bye": 0.0},
    "Go": {"win": 1.0, "draw": 0.5, "loss": 0.0, "bye": 0.0},
    "Tenis (partido)": {"win": 1.0, "draw": 0.0, "loss": 0.0, "bye": 0.0},
}

DEFAULT_RESULT_MAP: Dict[str, Tuple[str, str]] = {
    "1": ("win", "loss"),
    "0": ("loss", "win"),
    "0.5": ("draw", "draw"),
    "0,5": ("draw", "draw"),
    "1-0": ("win", "loss"),
    "0-1": ("loss", "win"),
    "0.5-0.5": ("draw", "draw"),
    "½-½": ("draw", "draw"),
}

# ---------------------------
# Core models
# ---------------------------
@dataclass
class Participant:
    pid: int
    name: str
    team_id: Optional[int] = None
    score: float = 0.0
    had_bye: bool = False
    opponents: List[int] = field(default_factory=list)
    per_round_points: List[float] = field(default_factory=list)  # para progresivo

@dataclass
class Team:
    team_id: int
    name: str
    player_ids: List[int]
    score: float = 0.0
    opponents: List[int] = field(default_factory=list)

@dataclass
class Game:
    a: Optional[int]   # jugador/seed/None
    b: Optional[int]
    result: str        # "1-0", "0.5-0.5", "2-1", "W" (walkover), etc.
    table: Optional[int] = None  # tablero/mesa
    meta: Dict[str, Any] = field(default_factory=dict)

# ---------------------------
# Scoring & Tie-breakers
# ---------------------------
class ScoringSystem:
    def __init__(self, name: str, points: Dict[str, float]):
        self.name = name
        # esperado: win/draw/loss/bye
        self.points = dict(points)

    def pts(self, outcome: str) -> float:
        return float(self.points.get(outcome, 0.0))

def tb_buchholz(players: Dict[int, Participant]) -> Dict[int, float]:
    # suma de puntuaciones finales de los rivales
    res = {}
    for p in players.values():
        res[p.pid] = sum(players[opp].score for opp in p.opponents if opp in players)
    return res

def tb_sonneborn_berger(players: Dict[int, Participant]) -> Dict[int, float]:
    # suma de (puntos del rival) por partidas ganadas + 0.5 * (puntos del rival) por tablas
    # Para calcular necesitamos saber por-partida; aproximamos desde per_round_points / opponents
    # Nota: en Swiss real guardarías resultados vs oponentes; aquí guardamos aproximado: si p ganó suma score rival; si empató suma 0.5*score rival.
    res = defaultdict(float)
    # Para hacerlo correcto, guardemos un registro per ronda en meta más abajo. Si no existe meta_detallada, usamos heurística 0.
    # Para este MVP, guardaremos en self.game_log por torneo; aquí solo definimos firma.
    return dict(res)

def tb_progressive(players: Dict[int, Participant]) -> Dict[int, float]:
    res = {}
    for p in players.values():
        prog = 0.0
        running = 0.0
        for r in p.per_round_points:
            running += r
            prog += running
        res[p.pid] = prog
    return res

# ---------------------------
# Persistencia
# ---------------------------
class StateStore:
    @staticmethod
    def save(path: str, data: Dict[str, Any]) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def load(path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

# ---------------------------
# Base Tournament
# ---------------------------
class TournamentBase:
    def __init__(self, mode: str = "Individual", game: str = "Ajedrez"):
        assert mode in ("Individual", "Equipos")
        self.mode = mode
        self.game = game
        self.scoring = ScoringSystem(game, SCORING_PRESETS[game])
        self.result_map = {k.lower(): v for k, v in DEFAULT_RESULT_MAP.items()}

        self.players: Dict[int, Participant] = {}
        self.teams: Dict[int, Team] = {}
        self.round_number: int = 0
        self.pairings: Any = []  # depende del tipo
        self.bye_id: Optional[int] = None
        self.results_all: List[List[Game]] = []
        self.game_log: List[Dict[str, Any]] = []  # para tiebreakers avanzados

    # --- comunes ---
    def update_points(self, name: str):
        if name not in SCORING_PRESETS:
            raise ValueError("Sistema no soportado")
        self.game = name
        self.scoring = ScoringSystem(name, SCORING_PRESETS[name])

    def update_result_map(self, mapping: Dict[str, Tuple[str, str]]):
        self.result_map = {str(k).strip().lower(): tuple(v) for k, v in mapping.items()}

    def canonical_outcome(self, raw: str) -> Optional[Tuple[str, str]]:
        return self.result_map.get(str(raw).strip().lower())

    def add_players(self, names: List[str]):
        start = len(self.players) + 1
        for i, nm in enumerate(names, start=1):
            pid = start + i - 1
            self.players[pid] = Participant(pid, nm)

    def add_teams(self, teams: List[Tuple[str, List[str]]]):
        # reset para coherencia (diseño sencillo)
        self.players.clear()
        self.teams.clear()
        pid = 1
        for tid, (tname, members) in enumerate(teams, start=1):
            pids = []
            for m in members:
                self.players[pid] = Participant(pid, m, team_id=tid)
                pids.append(pid); pid += 1
            self.teams[tid] = Team(tid, tname, pids)

    # --- API que cada subtipo debe implementar ---
    def new_round(self) -> List[str]:
        raise NotImplementedError

    def save_results(self, text: str) -> str:
        raise NotImplementedError

    def standings(self) -> List[str]:
        raise NotImplementedError

    # --- util común ---
    def export_csv(self, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = os.path.join(OUTPUT_DIR, "clasificacion.csv")
        with open(filename, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([f"Torneo: {self.game} — Modo: {self.mode} — Tipo: {type(self).__name__}"])
            wr.writerow([])
            wr.writerow(["Pos", "Nombre/Equipo", "Puntos"])
            rank = self.standings()
            for i, line in enumerate(rank, 1):
                wr.writerow([i, line])
        return filename

    def export_pdf(self, title: str, lines: List[str], filename: str) -> str:
        path = os.path.join(OUTPUT_DIR, filename)
        c = canvas.Canvas(path, pagesize=A4)
        w, h = A4
        margin = 2 * cm
        y = h - margin
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(w / 2, y, title)
        y -= 1.2 * cm
        c.setFont("Helvetica", 11)
        for line in lines:
            if y < margin:
                c.showPage()
                y = h - margin
                c.setFont("Helvetica", 11)
            c.drawString(margin, y, line)
            y -= 0.7 * cm
        c.save()
        return path

    def to_state(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "game": self.game,
            "scoring": self.scoring.points,
            "result_map": self.result_map,
            "players": [asdict(p) for p in self.players.values()],
            "teams": [asdict(t) for t in self.teams.values()],
            "round_number": self.round_number,
            "pairings": self.pairings,
            "bye_id": self.bye_id,
            "results_all": [[asdict(g) for g in ronda] for ronda in self.results_all],
            "type": type(self).__name__,
            "game_log": self.game_log,
        }

    @classmethod
    def from_state(cls, data: Dict[str, Any]) -> "TournamentBase":
        type_map = {"SwissTournament": SwissTournament, "LeagueTournament": LeagueTournament, "KOTournament": KOTournament}
        tclass = type_map.get(data.get("type", "SwissTournament"), SwissTournament)
        obj = tclass(mode=data.get("mode", "Individual"), game=data.get("game", "Ajedrez"))
        obj.scoring = ScoringSystem(obj.game, data.get("scoring", SCORING_PRESETS[obj.game]))
        obj.result_map = {k.lower(): tuple(v) for k, v in data.get("result_map", DEFAULT_RESULT_MAP).items()}
        obj.players = {p["pid"]: Participant(**p) for p in data.get("players", [])}
        obj.teams = {t["team_id"]: Team(**t) for t in data.get("teams", [])}
        obj.round_number = data.get("round_number", 0)
        obj.pairings = data.get("pairings", [])
        obj.bye_id = data.get("bye_id")
        obj.results_all = []
        for ronda in data.get("results_all", []):
            games = [Game(**g) for g in ronda]
            obj.results_all.append(games)
        obj.game_log = data.get("game_log", [])
        return obj

# ---------------------------
# Swiss Tournament
# ---------------------------
class SwissTournament(TournamentBase):
    def __init__(self, mode: str = "Individual", game: str = "Ajedrez"):
        super().__init__(mode, game)

    def _sorted_players(self) -> List[Participant]:
        return sorted(self.players.values(), key=lambda p: (-p.score, p.pid))

    def new_round(self) -> List[str]:
        self.round_number += 1
        lines = []
        # ordenar por puntos
        sorted_p = self._sorted_players()
        bye = None
        if len(sorted_p) % 2 == 1:
            # peor sin bye
            for cand in reversed(sorted_p):
                if not cand.had_bye:
                    bye = cand
                    break
            if bye:
                bye.had_bye = True
                bye_pts = self.scoring.pts("bye")
                bye.score += bye_pts
                bye.per_round_points.append(bye_pts)
                sorted_p = [p for p in sorted_p if p.pid != bye.pid]

        pairings = []
        for i in range(0, len(sorted_p), 2):
            a, b = sorted_p[i], sorted_p[i+1]
            a.opponents.append(b.pid)
            b.opponents.append(a.pid)
            pairings.append((a.pid, b.pid))
        self.pairings = pairings
        self.bye_id = bye.pid if bye else None

        for i, (a, b) in enumerate(pairings, start=1):
            lines.append(f"Mesa {i}: {self.players[a].name} vs {self.players[b].name}")
        if bye:
            lines.append(f"Bye: {bye.name} (+{self.scoring.pts('bye')} pts)")
        return lines

    def save_results(self, text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) != len(self.pairings):
            return f"Se esperaban {len(self.pairings)} resultados, llegaron {len(lines)}."
        round_games: List[Game] = []
        per_round_pts: Dict[int, float] = defaultdict(float)

        for (a_id, b_id), raw in zip(self.pairings, lines):
            m = self.canonical_outcome(raw)
            if m is None:
                return f"Resultado inválido '{raw}' para {self.players[a_id].name} vs {self.players[b_id].name}"
            o1, o2 = m
            pts1, pts2 = self.scoring.pts(o1), self.scoring.pts(o2)
            self.players[a_id].score += pts1
            self.players[b_id].score += pts2
            per_round_pts[a_id] += pts1
            per_round_pts[b_id] += pts2

            canonical = "1-0" if (o1, o2)==("win","loss") else \
                        "0-1" if (o1, o2)==("loss","win") else \
                        "0.5-0.5" if (o1, o2)==("draw","draw") else f"{pts1}-{pts2}"
            round_games.append(Game(a_id, b_id, canonical))

        # registrar progresivo
        for p in self.players.values():
            p.per_round_points.append(per_round_pts.get(p.pid, 0.0))

        self.results_all.append(round_games)
        self.game_log.append({"round": self.round_number, "games": [asdict(g) for g in round_games]})
        return "Resultados guardados."

    def standings(self) -> List[str]:
        # tie-breakers: Buchholz > Progresivo (SB placeholder en 0 hasta que registremos por-partida)
        players = self.players
        tb1 = tb_buchholz(players)
        tb2 = tb_progressive(players)
        ranked = sorted(players.values(),
                        key=lambda p: (-p.score, -tb1.get(p.pid, 0.0), -tb2.get(p.pid, 0.0), p.pid))
        return [f"{i+1}. {p.name} - {p.score:.2f} pts (Buchholz {tb1.get(p.pid,0):.2f}, Prog {tb2.get(p.pid,0):.2f})"
                for i, p in enumerate(ranked)]

# ---------------------------
# League (Round Robin)
# ---------------------------
class LeagueTournament(TournamentBase):
    def __init__(self, mode: str = "Individual", game: str = "Ajedrez"):
        super().__init__(mode, game)
        self.league_schedule: List[List[Tuple[int, int]]] = []  # lista de rondas

    def _generate_schedule(self):
        ids = list(self.players.keys())
        if len(ids) < 2:
            self.league_schedule = []
            return
        # Método del círculo
        if len(ids) % 2 == 1:
            ids.append(None)  # bye virtual
        n = len(ids)
        rounds = n - 1
        half = n // 2
        home = ids[:half]
        away = ids[half:]
        schedule = []
        for r in range(rounds):
            pairs = []
            for i in range(half):
                a, b = home[i], away[-(i+1)]
                if a is None or b is None:
                    # asignar bye real si corresponde
                    pid = a if b is None else b
                    if pid is not None:
                        # sumar bye aquí (opcional). Para consistencia, añadimos como emparejamiento None.
                        pairs.append((pid, None))
                else:
                    pairs.append((a, b))
            # rotación
            away.insert(0, home.pop(1))
            home.append(away.pop())
            schedule.append(pairs)
        self.league_schedule = schedule

    def new_round(self) -> List[str]:
        if not self.league_schedule:
            self._generate_schedule()
        if self.round_number >= len(self.league_schedule):
            return ["La liga ha finalizado."]
        self.round_number += 1
        rnd = self.league_schedule[self.round_number - 1]
        self.pairings = rnd
        lines = []
        mesa = 1
        for a, b in rnd:
            if b is None:
                # bye
                self.players[a].had_bye = True
                bye_pts = self.scoring.pts("bye")
                self.players[a].score += bye_pts
                self.players[a].per_round_points.append(bye_pts)
                lines.append(f"Bye: {self.players[a].name} (+{bye_pts} pts)")
            else:
                self.players[a].opponents.append(b)
                self.players[b].opponents.append(a)
                lines.append(f"Mesa {mesa}: {self.players[a].name} vs {self.players[b].name}")
                mesa += 1
        return lines

    def save_results(self, text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # contamos solo partidas reales (no byes)
        real_pairs = [(a,b) for (a,b) in self.pairings if b is not None]
        if len(lines) != len(real_pairs):
            return f"Se esperaban {len(real_pairs)} resultados (byes no se reportan), llegaron {len(lines)}."
        round_games: List[Game] = []
        per_round_pts: Dict[int, float] = defaultdict(float)
        for (a_id, b_id), raw in zip(real_pairs, lines):
            m = self.canonical_outcome(raw)
            if m is None:
                return f"Resultado inválido '{raw}' para {self.players[a_id].name} vs {self.players[b_id].name}"
            o1, o2 = m
            pts1, pts2 = self.scoring.pts(o1), self.scoring.pts(o2)
            self.players[a_id].score += pts1
            self.players[b_id].score += pts2
            per_round_pts[a_id] += pts1
            per_round_pts[b_id] += pts2
            canonical = "1-0" if (o1,o2)==("win","loss") else \
                        "0-1" if (o1,o2)==("loss","win") else \
                        "0.5-0.5" if (o1,o2)==("draw","draw") else f"{pts1}-{pts2}"
            round_games.append(Game(a_id, b_id, canonical))

        for p in self.players.values():
            self.players[p.pid].per_round_points.append(per_round_pts.get(p.pid, 0.0))
        self.results_all.append(round_games)
        return "Resultados guardados."

    def standings(self) -> List[str]:
        # Desempates típicos liga: diferencia (si existieran goles), aquí usamos Buchholz como fuerza de calendario
        players = self.players
        tb1 = tb_buchholz(players)
        ranked = sorted(players.values(), key=lambda p: (-p.score, -tb1.get(p.pid, 0.0), p.pid))
        return [f"{i+1}. {p.name} - {p.score:.2f} pts (Buchholz {tb1.get(p.pid,0):.2f})"
                for i, p in enumerate(ranked)]

# ---------------------------
# KO (Eliminación simple)
# ---------------------------
class KOTournament(TournamentBase):
    """
    Bracket simple. Cada 'ronda' son emparejamientos restantes.
    Entrada de resultados: una línea por match en orden mostrado. '1', '0' o '1-0'/'0-1'.
    """
    def __init__(self, mode: str = "Individual", game: str = "Ajedrez"):
        super().__init__(mode, game)
        self.bracket: List[List[Tuple[Optional[int], Optional[int]]]] = []  # rondas
        self.current_round_idx: int = 0

    def _seed_players(self):
        seeds = list(self.players.keys())
        random.shuffle(seeds)  # en real: usa ranking/ratings
        # completar a potencia de 2 con byes None
        n = 1
        while n < len(seeds): n *= 2
        while len(seeds) < n: seeds.append(None)
        # construir ronda 1
        round1 = []
        for i in range(0, len(seeds), 2):
            round1.append((seeds[i], seeds[i+1]))
        self.bracket = [round1]
        self.current_round_idx = 0

    def new_round(self) -> List[str]:
        if not self.bracket:
            self._seed_players()
        # si ronda actual está completa, ya transitamos con save_results()
        pairs = self.bracket[self.current_round_idx]
        self.pairings = pairs
        self.round_number = self.current_round_idx + 1
        lines = []
        mesa = 1
        next_pairs = []
        for a, b in pairs:
            if a is None and b is None:
                continue
            if a is None or b is None:
                # pase automático
                winner = a if b is None else b
                self.players[winner].score += self.scoring.pts("win")  # opcional
                lines.append(f"Bye avance: {self.players[winner].name}")
                # se resolverá al crear la siguiente ronda
            else:
                self.players[a].opponents.append(b); self.players[b].opponents.append(a)
                lines.append(f"Match {mesa}: {self.players[a].name} vs {self.players[b].name}")
                mesa += 1
        return lines

    def save_results(self, text: str) -> str:
        pairs = [(a,b) for (a,b) in self.pairings if a is not None and b is not None]
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) != len(pairs):
            return f"Se esperaban {len(pairs)} resultados, llegaron {len(lines)}."
        winners: List[Optional[int]] = []
        round_games: List[Game] = []
        for (a_id, b_id), raw in zip(pairs, lines):
            m = self.canonical_outcome(raw)
            if m is None:
                return f"Resultado inválido '{raw}' para {self.players[a_id].name} vs {self.players[b_id].name}"
            o1, o2 = m
            if o1 == "win" and o2 == "loss":
                winner = a_id; canonical = "1-0"
            elif o2 == "win" and o1 == "loss":
                winner = b_id; canonical = "0-1"
            else:
                # en KO no debería haber empates; si ocurre, decidir por desempate 'win' preferente a a_id
                winner = a_id if self.scoring.pts(o1) >= self.scoring.pts(o2) else b_id
                canonical = f"{self.scoring.pts(o1)}-{self.scoring.pts(o2)}"
            # sumar puntos opcional: win/loss
            self.players[a_id].score += self.scoring.pts(o1)
            self.players[b_id].score += self.scoring.pts(o2)
            winners.append(winner)
            round_games.append(Game(a_id, b_id, canonical))

        self.results_all.append(round_games)
        # construir siguiente ronda
        next_round: List[Tuple[Optional[int], Optional[int]]] = []
        # agregar también los byes automáticos de la ronda (jugadores que no tuvieron oponente)
        auto_advancers = [a if b is None else b for (a,b) in self.pairings if (a is None) ^ (b is None)]
        winners_iter = iter(winners + auto_advancers)
        chunk = list(winners_iter)
        # emparejar en parejas
        for i in range(0, len(chunk), 2):
            p1 = chunk[i]
            p2 = chunk[i+1] if i+1 < len(chunk) else None
            next_round.append((p1, p2))
        if len(next_round) == 1 and (next_round[0][1] is None):
            # campeón directo
            champion = next_round[0][0]
            return f"Campeón: {self.players[champion].name}"
        # mover a la siguiente ronda
        self.current_round_idx += 1
        if len(self.bracket) <= self.current_round_idx:
            self.bracket.append(next_round)
        else:
            self.bracket[self.current_round_idx] = next_round
        return "Resultados guardados. Ronda creada."

    def standings(self) -> List[str]:
        # En KO, el ranking final se decide al final; durante proceso, mostramos por rondas ganadas (score acumulado)
        ranked = sorted(self.players.values(), key=lambda p: (-p.score, p.pid))
        return [f"{i+1}. {p.name} - {p.score:.2f} pts" for i, p in enumerate(ranked)]

# ---------------------------
# Controlador + UI
# ---------------------------
class Controller:
    def __init__(self):
        self.t: TournamentBase = SwissTournament()

    def set_tournament(self, ttype: str, mode: str, game: str) -> str:
        type_map = {
            "Suizo": SwissTournament,
            "Liga": LeagueTournament,
            "Eliminación": KOTournament
        }
        cls = type_map.get(ttype, SwissTournament)
        self.t = cls(mode=mode, game=game)
        return f"Tipo={ttype}, Modo={mode}, Juego={game}"

    def save_state(self) -> str:
        data = self.t.to_state()
        path = StateStore.save(STATE_FILE, data)
        return f"Estado guardado en {path}"

    def load_state(self) -> str:
        data = StateStore.load(STATE_FILE)
        if not data:
            return "No hay estado guardado."
        self.t = TournamentBase.from_state(data)
        return f"Estado cargado ({type(self.t).__name__}, Ronda {self.t.round_number})."

C = Controller()

# ---- Helpers UI ----
def parse_players_text(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

def parse_teams_text(text: str) -> List[Tuple[str, List[str]]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        if ":" not in ln: continue
        team_name, players_str = ln.split(":", 1)
        players = [p.strip() for p in players_str.split(",") if p.strip()]
        out.append((team_name.strip(), players))
    return out

# ---- UI Actions ----
def ui_apply(ttype: str, mode: str, game: str):
    s1 = C.set_tournament(ttype, mode, game)
    return s1

def ui_set_points(game: str):
    C.t.update_points(game)
    return f"Sistema: {game} => {C.t.scoring.points}"

def ui_update_map(text_json: str):
    try:
        mapping = json.loads(text_json)
        for k, v in mapping.items():
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                return "Formato inválido: cada valor debe ser [outcomeA, outcomeB]"
        C.t.update_result_map({k: tuple(v) for k,v in mapping.items()})
        return "Mapeos actualizados."
    except Exception as e:
        return f"Error JSON: {e}"

def ui_add_players(text: str):
    names = parse_players_text(text)
    C.t.add_players(names)
    return f"Añadidos {len(names)} jugadores (total {len(C.t.players)})."

def ui_add_teams(text: str):
    parsed = parse_teams_text(text)
    C.t.add_teams(parsed)
    return f"Añadidos {len(C.t.teams)} equipos con {len(C.t.players)} jugadores."

def ui_new_round(_=None):
    lines = C.t.new_round()
    return "\n".join(lines)

def ui_save_results(text: str):
    msg = C.t.save_results(text)
    if "guardados" not in msg and "Campeón" not in msg:
        return msg
    st = "\n".join(C.t.standings())
    pdf = C.t.export_pdf(f"Clasificación - {type(C.t).__name__} - Ronda {C.t.round_number}",
                         C.t.standings(), f"standings_r{C.t.round_number}.pdf")
    return f"{msg}\n\n{st}\n\nPDF: {pdf}"

def ui_show_standings(_=None):
    return "\n".join(C.t.standings())

def ui_export_csv(_=None):
    return C.t.export_csv()

def ui_save_state(_=None):
    return C.save_state()

def ui_load_state(_=None):
    return C.load_state()

def ui_clear(_=None):
    C.__init__()
    return "Sesión reiniciada."

# ---------------------------
# Interfaz Gradio
# ---------------------------
with gr.Blocks(title="Universal Competition Tracker") as demo:
    gr.Markdown("# 🏆 Universal Competition Tracker\n**Swiss · League · KO** — Individual & Equipos")

    with gr.Row():
        ttype = gr.Dropdown(["Suizo", "Liga", "Eliminación"], value="Suizo", label="Tipo de torneo")
        mode = gr.Radio(["Individual", "Equipos"], value="Individual", label="Modo")
        game = gr.Dropdown(list(SCORING_PRESETS.keys()), value="Ajedrez", label="Juego / Sistema de puntuación")
        btn_apply = gr.Button("Crear/Reset")
    status = gr.Textbox(label="Estado", interactive=False)

    btn_apply.click(ui_apply, [ttype, mode, game], status)

    with gr.Tab("Puntuación / Result Map"):
        with gr.Row():
            dd_game = gr.Dropdown(list(SCORING_PRESETS.keys()), value="Ajedrez", label="Preset de puntos")
            btn_points = gr.Button("Aplicar preset")
            out_points = gr.Textbox(label="Puntos activos", interactive=False)
        btn_points.click(ui_set_points, dd_game, out_points)

        gr.Markdown("Edite el mapeo entrada→outcomes. Ej: `{ \"1\": [\"win\",\"loss\"], \"0.5\": [\"draw\",\"draw\"] }`")
        txt_map = gr.Textbox(lines=6, label="Result map (JSON)")
        btn_map = gr.Button("Actualizar map")
        out_map = gr.Textbox(label="Estado map", interactive=False)
        btn_map.click(ui_update_map, txt_map, out_map)

    with gr.Tab("Participantes"):
        with gr.Row():
            txt_players = gr.Textbox(lines=6, label="Jugadores (uno por línea)")
            btn_add_players = gr.Button("Añadir jugadores")
            out_add_players = gr.Textbox(label="Estado jugadores", interactive=False)
        btn_add_players.click(ui_add_players, txt_players, out_add_players)

        gr.Markdown("Formato equipos: `Equipo: Jug1, Jug2, Jug3`")
        txt_teams = gr.Textbox(lines=6, label="Equipos (reemplaza jugadores)")
        btn_add_teams = gr.Button("Añadir equipos")
        out_add_teams = gr.Textbox(label="Estado equipos", interactive=False)
        btn_add_teams.click(ui_add_teams, txt_teams, out_add_teams)

    with gr.Tab("Rondas"):
        btn_round = gr.Button("Generar nueva ronda / mostrar emparejamientos")
        out_round = gr.Textbox(label="Emparejamientos", interactive=False)
        btn_round.click(ui_new_round, None, out_round)

    with gr.Tab("Resultados"):
        gr.Markdown("Suizo/Liga: una línea por partida en orden. KO: una línea por match.\nValores válidos: `1`, `0`, `0.5`, `1-0`, `0-1`, `0.5-0.5`, etc.")
        txt_results = gr.Textbox(lines=10, label="Resultados")
        btn_save_results = gr.Button("Guardar resultados")
        out_results = gr.Textbox(label="Salida", interactive=False)
        btn_save_results.click(ui_save_results, txt_results, out_results)

    with gr.Tab("Clasificación / Export"):
        btn_stand = gr.Button("Mostrar standings")
        out_stand = gr.Textbox(label="Standings", interactive=False)
        btn_stand.click(ui_show_standings, None, out_stand)

        btn_csv = gr.Button("Exportar CSV")
        out_csv = gr.Textbox(label="Ruta CSV", interactive=False)
        btn_csv.click(ui_export_csv, None, out_csv)

    with gr.Tab("Estado"):
        with gr.Row():
            btn_save = gr.Button("Guardar estado")
            out_save = gr.Textbox()
            btn_load = gr.Button("Cargar estado")
            out_load = gr.Textbox()
            btn_reset = gr.Button("Reset")
            out_reset = gr.Textbox()
        btn_save.click(ui_save_state, None, out_save)
        btn_load.click(ui_load_state, None, out_load)
        btn_reset.click(ui_clear, None, out_reset)

if __name__ == "__main__":
    # Para acceso LAN: server_name="0.0.0.0"
    demo.launch()
