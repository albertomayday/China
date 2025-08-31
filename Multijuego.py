# -*- coding: utf-8 -*-
"""
Gestor universal de torneos suizos - individual y por equipos
Soporta múltiples sistemas de puntuación configurables por juego.
Genera PDFs de clasificación con ReportLab y UI con Gradio.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
import os
import json
import random
import csv
import gradio as gr
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# ---------------------------
# Config y constantes
# ---------------------------
OUTPUT_DIR = "./torneo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sistemas de puntuación predefinidos (puedes añadir más)
SCORING_SYSTEMS: Dict[str, Dict[str, float]] = {
    "Ajedrez": {"win": 1.0, "draw": 0.5, "loss": 0.0, "bye": 1.0},
    "Fútbol": {"win": 3.0, "draw": 1.0, "loss": 0.0, "bye": 0.0},
    "Go": {"win": 1.0, "draw": 0.5, "loss": 0.0, "bye": 0.0},
    "Tenis (por partido)": {"win": 1.0, "draw": 0.0, "loss": 0.0, "bye": 0.0},
}

# Mapeo de entrada a outcomes; se puede editar desde la UI si quieres
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

STATE_FILE = os.path.join(OUTPUT_DIR, "tournament_state.json")

# ---------------------------
# Data classes
# ---------------------------
@dataclass
class Player:
    player_id: int
    name: str
    team_id: Optional[int] = None
    score: float = 0.0
    opponents: List[int] = None
    colors: List[str] = None
    had_bye: bool = False

    def __post_init__(self):
        if self.opponents is None:
            self.opponents = []
        if self.colors is None:
            self.colors = []

@dataclass
class Team:
    team_id: int
    name: str
    player_ids: List[int]
    score: float = 0.0
    opponents: List[int] = None

    def __post_init__(self):
        if self.opponents is None:
            self.opponents = []

@dataclass
class Game:
    white: Optional[int]
    black: Optional[int]
    result: str  # canonical, e.g., "1-0", "0.5-0.5" or "2-1" for team-sum

# ---------------------------
# Tournament core
# ---------------------------
class Tournament:
    def __init__(self):
        self.mode: str = "Individual"  # "Individual" | "Equipos"
        self.game_type: str = "Ajedrez"
        self.points = SCORING_SYSTEMS[self.game_type].copy()
        self.result_map = {k.lower(): v for k, v in DEFAULT_RESULT_MAP.items()}
        self.players: List[Player] = []
        self.teams: List[Team] = []
        self.round_number: int = 0
        self.pairings = []  # structure depends on mode
        self.bye_id: Optional[int] = None
        self.results_all: List[List[Game]] = []  # por ronda, lista de Game

    # ---- configuración ----
    def set_mode(self, mode: str):
        assert mode in ("Individual", "Equipos")
        self.mode = mode

    def set_game_type(self, game_type: str):
        if game_type not in SCORING_SYSTEMS:
            raise ValueError("Juego no soportado")
        self.game_type = game_type
        self.points = SCORING_SYSTEMS[game_type].copy()

    def update_result_map(self, mapping: Dict[str, Tuple[str, str]]):
        # normalizar claves
        self.result_map = {str(k).strip().lower(): tuple(v) for k, v in mapping.items()}

    # ---- persistencia simple ----
    def save_state(self, path: str = STATE_FILE):
        state = {
            "mode": self.mode,
            "game_type": self.game_type,
            "points": self.points,
            "result_map": self.result_map,
            "players": [asdict(p) for p in self.players],
            "teams": [asdict(t) for t in self.teams],
            "round_number": self.round_number,
            "pairings": self.pairings,
            "bye_id": self.bye_id,
            "results_all": [[asdict(g) for g in ronda] for ronda in self.results_all],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        return path

    def load_state(self, path: str = STATE_FILE) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.mode = state.get("mode", self.mode)
        self.set_game_type(state.get("game_type", self.game_type))
        self.result_map = {k.lower(): tuple(v) for k, v in state.get("result_map", {}).items()}
        self.players = [Player(**p) for p in state.get("players", [])]
        self.teams = [Team(**t) for t in state.get("teams", [])]
        self.round_number = state.get("round_number", 0)
        self.pairings = state.get("pairings", [])
        self.bye_id = state.get("bye_id")
        self.results_all = []
        for ronda in state.get("results_all", []):
            lista = []
            for g in ronda:
                lista.append(Game(**g))
            self.results_all.append(lista)
        return True

    # ---- util ----
    def find_player(self, pid: int) -> Optional[Player]:
        for p in self.players:
            if p.player_id == pid:
                return p
        return None

    def find_team(self, tid: int) -> Optional[Team]:
        for t in self.teams:
            if t.team_id == tid:
                return t
        return None

    # ---- emparejamientos individual ----
    def generate_pairings_individual(self) -> Tuple[List[Tuple[int, int]], Optional[int]]:
        sorted_p = sorted(self.players, key=lambda p: (-p.score, p.player_id))
        bye = None
        if len(sorted_p) % 2 == 1:
            # asigna bye al peor que no haya tenido bye
            candidates = [p for p in reversed(sorted_p) if not p.had_bye]
            if candidates:
                bye = candidates[0]
                bye.had_bye = True
                bye.score += float(self.points.get("bye", 0.0))
                sorted_p = [p for p in sorted_p if p.player_id != bye.player_id]
        pairings = []
        for i in range(0, len(sorted_p), 2):
            p1, p2 = sorted_p[i], sorted_p[i+1]
            p1.opponents.append(p2.player_id)
            p2.opponents.append(p1.player_id)
            pairings.append((p1.player_id, p2.player_id))
        self.pairings = pairings
        self.bye_id = bye.player_id if bye else None
        return pairings, bye.player_id if bye else None

    # ---- emparejamientos por equipos ----
    def generate_pairings_teams(self) -> Tuple[List[Tuple[int, int, List[Tuple[int, int]]]], Optional[int]]:
        sorted_t = sorted(self.teams, key=lambda t: (-t.score, t.team_id))
        bye = None
        if len(sorted_t) % 2 == 1:
            candidates = [t for t in reversed(sorted_t)]
            bye = candidates[0]
            bye.score += float(self.points.get("bye", 0.0))
            sorted_t = [t for t in sorted_t if t.team_id != bye.team_id]
        pairings = []
        for i in range(0, len(sorted_t), 2):
            t1, t2 = sorted_t[i], sorted_t[i+1]
            t1.opponents.append(t2.team_id)
            t2.opponents.append(t1.team_id)
            # emparejar jugadores por índice (si distinto tamaño, trunca o rellena con None)
            n = max(len(t1.player_ids), len(t2.player_ids))
            matches = []
            for j in range(n):
                a = t1.player_ids[j] if j < len(t1.player_ids) else None
                b = t2.player_ids[j] if j < len(t2.player_ids) else None
                matches.append((a, b))
            pairings.append((t1.team_id, t2.team_id, matches))
        self.pairings = pairings
        self.bye_id = bye.team_id if bye else None
        return pairings, bye.team_id if bye else None

    # ---- normalizar input de resultado a outcomes ----
    def canonical_outcome(self, raw: str) -> Optional[Tuple[str, str]]:
        k = str(raw).strip().lower()
        return self.result_map.get(k)

    # ---- guardar resultados individuales ----
    def save_results_individual(self, pairings: List[Tuple[int,int]], results: List[str]) -> Tuple[bool, str]:
        if len(pairings) != len(results):
            return False, f"Se esperaban {len(pairings)} resultados, llegaron {len(results)}."
        round_games = []
        for (p1_id, p2_id), raw in zip(pairings, results):
            mapping = self.canonical_outcome(raw)
            p1 = self.find_player(p1_id)
            p2 = self.find_player(p2_id)
            if mapping is None:
                return False, f"Resultado inválido '{raw}' para {p1.name} vs {p2.name}"
            o1, o2 = mapping
            pts1 = float(self.points.get(o1, 0.0))
            pts2 = float(self.points.get(o2, 0.0))
            p1.score += pts1
            p2.score += pts2
            # canonical store
            if o1 == "win" and o2 == "loss":
                canonical = "1-0"
            elif o1 == "loss" and o2 == "win":
                canonical = "0-1"
            elif o1 == "draw" and o2 == "draw":
                canonical = "0.5-0.5"
            else:
                canonical = f"{pts1}-{pts2}"
            white = p1_id
            black = p2_id
            round_games.append(Game(white, black, canonical))
        self.results_all.append(round_games)
        return True, "Resultados individuales guardados."

    # ---- guardar resultados por equipos (lista de bloques por match) ----
    def save_results_teams(self, pairings: List[Tuple[int,int,List[Tuple[int,int]]]], results_blocks: List[List[str]]) -> Tuple[bool, str]:
        if len(pairings) != len(results_blocks):
            return False, f"Se esperaban {len(pairings)} bloques de resultados (uno por match), llegaron {len(results_blocks)}."
        round_games = []
        for (t1_id, t2_id, matches), block in zip(pairings, results_blocks):
            team1 = self.find_team(t1_id)
            team2 = self.find_team(t2_id)
            # cada elemento de block es resultado de un tablero en order matches
            if len(block) != len(matches):
                return False, f"Match {team1.name} vs {team2.name}: se esperaban {len(matches)} resultados de tablero, llegaron {len(block)}."
            team_pts1 = 0.0
            team_pts2 = 0.0
            for (a_id, b_id), raw in zip(matches, block):
                # si a_id o b_id es None (jugador faltante), tratar como derrota/bye según configuración
                if a_id is None and b_id is None:
                    continue
                mapping = self.canonical_outcome(raw)
                if mapping is None:
                    return False, f"Resultado inválido '{raw}' en match {team1.name} vs {team2.name}"
                o1, o2 = mapping
                pts1 = float(self.points.get(o1, 0.0))
                pts2 = float(self.points.get(o2, 0.0))
                if a_id:
                    p_a = self.find_player(a_id)
                    p_a.score += pts1
                if b_id:
                    p_b = self.find_player(b_id)
                    p_b.score += pts2
                team_pts1 += pts1
                team_pts2 += pts2
            # decidir puntos de match (configurable: aquí usamos same scale as win/draw/loss)
            if team_pts1 > team_pts2:
                team1.score += float(self.points.get("win", 1.0))
                team2.score += float(self.points.get("loss", 0.0))
            elif team_pts2 > team_pts1:
                team2.score += float(self.points.get("win", 1.0))
                team1.score += float(self.points.get("loss", 0.0))
            else:
                team1.score += float(self.points.get("draw", 0.0))
                team2.score += float(self.points.get("draw", 0.0))
            # guardamos un Game que describe el match score (por claridad)
            canonical = f"{team_pts1}-{team_pts2}"
            round_games.append(Game(None, None, canonical))
        self.results_all.append(round_games)
        return True, "Resultados por equipos guardados."

    # ---- standings ----
    def standings_individual(self) -> List[str]:
        ranking = sorted(self.players, key=lambda p: (-p.score, p.player_id))
        lines = []
        for i, p in enumerate(ranking, 1):
            lines.append(f"{i}. {p.name} - {p.score:.2f} pts")
        return lines

    def standings_teams(self) -> List[str]:
        ranking = sorted(self.teams, key=lambda t: (-t.score, t.team_id))
        lines = []
        for i, t in enumerate(ranking, 1):
            lines.append(f"{i}. {t.name} - {t.score:.2f} pts")
        return lines

    # ---- exportar CSV ----
    def export_csv(self, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = os.path.join(OUTPUT_DIR, "clasificacion.csv")
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([f"Torneo: {self.game_type} — Modo: {self.mode}"])
            writer.writerow([])
            if self.mode == "Individual":
                writer.writerow(["Pos", "Jugador", "Puntos"])
                for i, line in enumerate(self.standings_individual(), 1):
                    # line = "1. Nombre - x.xx pts"
                    writer.writerow([i, line])
            else:
                writer.writerow(["Pos", "Equipo", "Puntos"])
                for i, line in enumerate(self.standings_teams(), 1):
                    writer.writerow([i, line])
        return filename

    # ---- exportar PDF de standings ----
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
            y -= 0.8 * cm
        c.save()
        return path

# ---------------------------
# Instancia global y funciones UI
# ---------------------------
t = Tournament()
# tratar de cargar estado preexistente (no obligatorio)
t.load_state()

# ---- helpers UI (parsers) ----
def parse_players_text(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

def parse_teams_text(text: str) -> List[Tuple[str, List[str]]]:
    """
    Formato esperado por línea:
    Team Name: Player A, Player B, Player C
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        if ":" not in ln:
            continue
        team_name, players_str = ln.split(":", 1)
        players = [p.strip() for p in players_str.split(",") if p.strip()]
        out.append((team_name.strip(), players))
    return out

# ---- UI actions ----
def ui_set_mode(game_mode: str):
    t.set_mode(game_mode)
    t.save_state()
    return f"Modo: {t.mode}"

def ui_set_game(game_type: str):
    t.set_game_type(game_type)
    t.save_state()
    return f"Juego: {t.game_type} (puntos: {t.points})"

def ui_update_result_map(text_json: str):
    try:
        mapping = json.loads(text_json)
        # Validate values are 2-lists
        for k, v in mapping.items():
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                return "Formato inválido: cada valor debe ser una lista/tupla de 2 elementos."
        t.update_result_map({k: tuple(v) for k, v in mapping.items()})
        t.save_state()
        return "Mapeos actualizados."
    except Exception as e:
        return f"Error JSON: {e}"

def ui_add_players(text: str):
    names = parse_players_text(text)
    start = len(t.players) + 1
    for i, name in enumerate(names):
        t.players.append(Player(start + i, name))
    t.save_state()
    return f"Añadidos {len(names)} jugadores (total {len(t.players)})."

def ui_add_teams(text: str):
    parsed = parse_teams_text(text)
    # reset players and teams to avoid conflicts: decisión de diseño (puedes cambiar)
    t.players = []
    t.teams = []
    pid = 1
    for tid, (team_name, players) in enumerate(parsed, start=1):
        pids = []
        for nm in players:
            t.players.append(Player(pid, nm, team_id=tid))
            pids.append(pid)
            pid += 1
        t.teams.append(Team(team_id=tid, name=team_name, player_ids=pids))
    t.save_state()
    return f"Añadidos {len(t.teams)} equipos, {len(t.players)} jugadores."

def ui_new_round(_=None):
    t.round_number += 1
    if t.mode == "Individual":
        pairings, bye = t.generate_pairings_individual()
        lines = [f"Mesa {i+1}: {t.find_player(a).name} vs {t.find_player(b).name}" for i, (a,b) in enumerate(pairings)]
        if bye:
            lines.append(f"Bye: {t.find_player(bye).name} (+{t.points.get('bye',0)} pts)")
        t.save_state()
        return "\n".join(lines)
    else:
        pairings, bye = t.generate_pairings_teams()
        lines = []
        for i, (t1_id, t2_id, matches) in enumerate(pairings, start=1):
            t1 = t.find_team(t1_id); t2 = t.find_team(t2_id)
            lines.append(f"Match {i}: {t1.name} vs {t2.name}")
            for j, (a,b) in enumerate(matches, start=1):
                na = t.find_player(a).name if a else "---"
                nb = t.find_player(b).name if b else "---"
                lines.append(f"  Tablero {j}: {na} vs {nb}")
        if bye:
            lines.append(f"Bye equipo: {t.find_team(bye).name} (+{t.points.get('bye',0)} pts)")
        t.save_state()
        return "\n".join(lines)

def ui_save_results(text: str):
    """
    Individual: cada línea corresponde a un resultado en el mismo orden de emparejamientos
    Equipos: bloques separados por línea en blanco; cada bloque tiene resultados de los tableros en order.
    """
    if t.mode == "Individual":
        lines = parse_players_text(text)
        ok, msg = t.save_results_individual(t.pairings, lines)
        if not ok:
            return msg
        standings = t.standings_individual()
        pdf = t.export_pdf(f"Clasificación - Ronda {t.round_number}", standings, f"standings_indiv_r{t.round_number}.pdf")
        t.save_state()
        return "\n".join(standings) + f"\nPDF: {pdf}"
    else:
        # parse blocks
        raw_lines = text.splitlines()
        blocks = []
        cur = []
        for ln in raw_lines:
            if not ln.strip():
                if cur:
                    blocks.append([l.strip() for l in cur])
                    cur = []
            else:
                cur.append(ln)
        if cur:
            blocks.append([l.strip() for l in cur])
        ok, msg = t.save_results_teams(t.pairings, blocks)
        if not ok:
            return msg
        standings = t.standings_teams()
        pdf = t.export_pdf(f"Clasificación Equipos - Ronda {t.round_number}", standings, f"standings_teams_r{t.round_number}.pdf")
        t.save_state()
        return "\n".join(standings) + f"\nPDF: {pdf}"

def ui_show_standings(_=None):
    if t.mode == "Individual":
        return "\n".join(t.standings_individual())
    else:
        return "\n".join(t.standings_teams())

def ui_export_csv(_=None):
    path = t.export_csv()
    return path

def ui_save_state(_=None):
    p = t.save_state()
    return f"Estado guardado en {p}"

def ui_load_state(_=None):
    ok = t.load_state()
    return "Estado cargado." if ok else "No hay state guardado."

def ui_clear(_=None):
    t.__init__()  # reinicia
    t.save_state()
    return "Torneo reiniciado."

# ---------------------------
# Interfaz Gradio
# ---------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🏆 Torneo Universal — Individual y por Equipos (multijuego)")

    with gr.Row():
        mode = gr.Radio(["Individual", "Equipos"], value=t.mode, label="Modo")
        game = gr.Dropdown(list(SCORING_SYSTEMS.keys()), value=t.game_type, label="Juego / Sistema de puntuación")
        btn_apply = gr.Button("Aplicar")
    status = gr.Textbox(label="Estado", interactive=False)

    btn_apply.click(lambda m, g: (ui_set_mode(m), ui_set_game(g)), [mode, game], status)

    with gr.Tab("Configuración avanzada"):
        gr.Markdown("Puedes editar el mapeo de entradas a outcomes (JSON). Ejemplo: `{\"1\":[\"win\",\"loss\"], \"0\":[\"loss\",\"win\"], \"0.5\":[\"draw\",\"draw\"]}`")
        txt_map = gr.Textbox(lines=6, label="Result map (JSON)")
        btn_map = gr.Button("Actualizar map")
        out_map = gr.Textbox(label="Estado map")
        btn_map.click(ui_update_result_map, txt_map, out_map)

    with gr.Tab("Añadir participantes"):
        with gr.Column():
            txt_players = gr.Textbox(lines=6, label="Jugadores (uno por línea) — solo modo Individual")
            btn_add_players = gr.Button("Añadir jugadores")
            out_add_players = gr.Textbox(label="Estado jugadores")
            btn_add_players.click(ui_add_players, txt_players, out_add_players)

            txt_teams = gr.Textbox(lines=8, label="Equipos (formato: Equipo: Jug1, Jug2, Jug3 ... )")
            btn_add_teams = gr.Button("Añadir equipos (reemplaza los anteriores)")
            out_add_teams = gr.Textbox(label="Estado equipos")
            btn_add_teams.click(ui_add_teams, txt_teams, out_add_teams)

    with gr.Tab("Rondas"):
        btn_round = gr.Button("Generar nueva ronda")
        out_round = gr.Textbox(label="Emparejamientos")
        btn_round.click(ui_new_round, None, out_round)

    with gr.Tab("Resultados"):
        gr.Markdown("Individual: introduzca un resultado por línea, en el mismo orden que los emparejamientos.\n\nEquipos: introduzca bloques separados por línea en blanco; cada bloque contiene tantos resultados como tableros del match.")
        txt_results = gr.Textbox(lines=12, label="Resultados")
        btn_save_results = gr.Button("Guardar resultados y generar PDF")
        out_results = gr.Textbox(label="Salida")
        btn_save_results.click(ui_save_results, txt_results, out_results)

    with gr.Tab("Clasificación / Export"):
        btn_stand = gr.Button("Mostrar standings")
        out_stand = gr.Textbox(label="Standings")
        btn_stand.click(ui_show_standings, None, out_stand)
        btn_csv = gr.Button("Exportar CSV")
        out_csv = gr.Textbox(label="Archivo CSV")
        btn_csv.click(ui_export_csv, None, out_csv)

    with gr.Tab("Estado / Persistencia"):
        btn_save = gr.Button("Guardar estado")
        out_save = gr.Textbox()
        btn_save.click(ui_save_state, None, out_save)
        btn_load = gr.Button("Cargar estado")
        out_load = gr.Textbox()
        btn_load.click(ui_load_state, None, out_load)
        btn_reset = gr.Button("Reset completo")
        out_reset = gr.Textbox()
        btn_reset.click(ui_clear, None, out_reset)

if __name__ == "__main__":
    demo.launch()
