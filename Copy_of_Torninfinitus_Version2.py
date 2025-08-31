# -*- coding: utf-8 -*-
# Archivo convertido de Colab a .py
pip install gradio reportlab
import os
import random
import csv
import gradio as gr
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# ==== CONFIGURACIÓN DEL TORNEO ====
OUTPUT_DIR = "./ajedrez_torneo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TOURNAMENT_TITLE = "TORNEO DE AJEDREZ"
SUBTITLE = "Sistema Suizo"
RITMO = "Ritmo: 15 minutos"

# ==== CLASE JUGADOR ====
class Player:
    def __init__(self, player_id, name):
        self.player_id = player_id
        self.name = name
        self.score = 0.0
        self.opponents = []
        self.colors = []
        self.had_bye = False

players = []
round_number = 0
pairings = []
bye_player = None
results_all = []

# ==== FUNCIONES INTERNAS ====
def assign_bye(players_list):
    candidates = [p for p in sorted(players_list, key=lambda x: (x.score, x.player_id)) if not p.had_bye]
    if not candidates:
        return None
    bye = candidates[0]
    bye.had_bye = True
    bye.score += 1.0
    return bye

def choose_color(p1, p2):
    w1, b1 = p1.colors.count("W"), p1.colors.count("B")
    w2, b2 = p2.colors.count("W"), p2.colors.count("B")
    if w1 - b1 > w2 - b2:
        return "B", "W"
    elif w2 - b2 > w1 - b1:
        return "W", "B"
    return ("W", "B") if random.choice([True, False]) else ("B", "W")

def generate_pairings(players_list):
    sorted_p = sorted(players_list, key=lambda p: (-p.score, p.player_id))
    bye = None
    if len(sorted_p) % 2 == 1:
        bye = assign_bye(sorted_p)
        sorted_p = [p for p in sorted_p if p.player_id != bye.player_id]
    pr = []
    for i in range(0, len(sorted_p), 2):
        p1, p2 = sorted_p[i], sorted_p[i+1]
        c1, c2 = choose_color(p1, p2)
        p1.colors.append(c1); p2.colors.append(c2)
        p1.opponents.append(p2.player_id); p2.opponents.append(p1.player_id)
        pr.append((p1, p2, c1, c2))
    return pr, bye

# ==== FUNCIONES DE GENERACIÓN DE PDF ====
def crear_cartel_ronda(ronda, pairings=None, results=None, bye=None, standings_text=None):
    filename = os.path.join(OUTPUT_DIR, f"ronda_{ronda}.pdf")
    c = canvas.Canvas(filename, pagesize=A4)
    w, h = A4
    margin = 2*cm
    y_position = h - margin

    # --- Título del Torneo ---
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(w/2, y_position, TOURNAMENT_TITLE)
    y_position -= 1.5*cm
    c.setFont("Helvetica", 12)
    c.drawCentredString(w/2, y_position, f"{SUBTITLE} · {RITMO}")
    y_position -= 2*cm

    # --- Título de la Ronda ---
    c.setFont("Helvetica-Bold", 30)
    c.drawCentredString(w/2, y_position, f"RONDA {ronda}")
    y_position -= 3*cm

    # --- Resultados de la Ronda Anterior (si existen) ---
    if results:
        if y_position < margin + 2*cm:
            c.showPage()
            y_position = h - margin
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y_position, "Resultados Ronda Anterior:")
        y_position -= 1*cm
        c.setFont("Helvetica", 12)
        for r in results:
            if y_position < margin:
                c.showPage()
                c.setFont("Helvetica", 12)
                y_position = h - margin
            c.drawString(margin, y_position, r)
            y_position -= 0.8*cm
        y_position -= 1.5*cm

    # --- Clasificación Actual (si existe) ---
    if standings_text:
        if y_position < margin + 2*cm:
            c.showPage()
            y_position = h - margin
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y_position, "Clasificación Actual:")
        y_position -= 1*cm
        c.setFont("Courier", 10)
        for line in standings_text.splitlines():
            if y_position < margin:
                c.showPage()
                c.setFont("Courier", 10)
                y_position = h - margin
            c.drawString(margin, y_position, line)
            y_position -= 0.6*cm
        y_position -= 1.5*cm

    # --- Próximos Emparejamientos (si existen) ---
    if pairings:
        if y_position < margin + 2*cm:
            c.showPage()
            y_position = h - margin
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y_position, "Próximos Emparejamientos:")
        y_position -= 1*cm
        c.setFont("Helvetica", 12)
        for i, (p1,p2,c1,c2) in enumerate(pairings):
            if y_position < margin:
                c.showPage()
                c.setFont("Helvetica", 12)
                y_position = h - margin
            c.drawString(margin, y_position, f"Mesa {i+1}: {p1.name} ({c1}) vs {p2.name} ({c2})")
            y_position -= 0.8*cm
        if bye:
            if y_position < margin:
                c.showPage()
                c.setFont("Helvetica", 12)
                y_position = h - margin
            c.drawString(margin, y_position, f"Bye: {bye.name} (+1 punto)")
            y_position -= 1*cm

    c.save()
    return filename

# ==== DESEMPATES ====
def find_player(pid, players_list):
    for p in players_list:
        if p.player_id == pid: return p
    return None

def find_player_by_name(name, players_list):
    for p in players_list:
        if p.name == name: return p
    return None

def calculate_buchholz(player, players_list):
    return sum([find_player(op, players_list).score for op in player.opponents])

def calculate_sonneborn(player, players_list, results_history):
    sb = 0
    for r in results_history:
        for line in r:
            if player.name in line:
                if "1 - 0" in line and line.startswith(player.name):
                    opp = find_player_by_name(line.split()[-1], players_list)
                    if opp: sb += opp.score
                elif "0 - 1" in line and line.endswith(player.name):
                    opp = find_player_by_name(line.split()[0], players_list)
                    if opp: sb += opp.score
                elif "0.5" in line:
                    opp_name = line.split()[0] if line.split()[2] == player.name else line.split()[2]
                    opp = find_player_by_name(opp_name, players_list)
                    if opp: sb += opp.score * 0.5
    return sb

def calculate_progressive(player, results_history):
    total = 0
    running = 0
    for r in results_history:
        for line in r:
            if player.name in line:
                if "1 - 0" in line and line.startswith(player.name):
                    running += 1
                elif "0 - 1" in line and line.endswith(player.name):
                    running += 1
                elif "0.5" in line:
                    running += 0.5
        total += running
    return total

# ==== ORDENACIÓN FINAL (GANADOR ÚNICO) ====
def final_standings(players_list, results_history):
    return sorted(
        players_list,
        key=lambda p: (
            -p.score,
            -calculate_sonneborn(p, players_list, results_history),
            -calculate_buchholz(p, players_list),
            -calculate_progressive(p, results_history),
            random.random()
        )
    )

def export_csv(players_list, results_history):
    filename = os.path.join(OUTPUT_DIR, "clasificacion.csv")
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Jugador", "Puntos", "SB", "Buchholz", "Progresivo"])
        for p in final_standings(players_list, results_history):
            writer.writerow([p.name, p.score, calculate_sonneborn(p, players_list, results_history), calculate_buchholz(p, players_list), calculate_progressive(p, results_history)])
        writer.writerow([])
        writer.writerow(["Resultados por ronda"])
        for r, res in enumerate(results_history, 1):
            writer.writerow([f"Ronda {r}"])
            for line in res:
                writer.writerow([line])
    return filename

def clear_pdfs():
    """Borra todos los archivos PDF en el directorio de salida."""
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith(".pdf"):
            os.remove(os.path.join(OUTPUT_DIR, file))
    return "Archivos PDF anteriores borrados."

# ==== INTERFAZ GRADIO ====
def add_players_ui(names):
    global players, round_number, results_all
    clear_pdfs()
    players = [Player(i+1, name.strip()) for i,name in enumerate(names.splitlines()) if name.strip()]
    round_number = 0
    results_all = []
    return f"Jugadores cargados: {len(players)}"

def new_round_ui():
    global round_number, pairings, bye_player, results_all, players
    if not players: return "Primero añade jugadores."

    # Check if a previous round exists and generate PDF for it
    if round_number > 0:
        standings_text = standings_ui()
        previous_results = results_all[round_number - 1] if round_number - 1 < len(results_all) else None
        crear_cartel_ronda(round_number, results=previous_results, standings_text=standings_text, bye=bye_player)

    round_number += 1
    pairings, bye_player = generate_pairings(players)
    crear_cartel_ronda(round_number, pairings=pairings, bye=bye_player)

    text = f"RONDA {round_number}\n"
    for i,(p1,p2,c1,c2) in enumerate(pairings,1):
        text += f"Mesa {i}: {p1.name} ({c1}) vs {p2.name} ({c2})\n"
    if bye_player:
        text += f"Bye: {bye_player.name} (+1 punto)\n"
    return text

def save_results_ui(results_str):
    global results_all, players, pairings, bye_player
    if not pairings: return "No hay emparejamientos."
    results = []
    lines = results_str.splitlines()
    if len(lines) < len(pairings):
        return f"Error: Se esperaban {len(pairings)} líneas de resultado, pero se recibieron {len(lines)}. Introduce un resultado por cada mesa emparejada."

    for line, (p1,p2,c1,c2) in zip(lines, pairings):
        val = line.strip()
        if val == "1":
            p1.score += 1; results.append(f"{p1.name} 1 - 0 {p2.name}")
        elif val == "0":
            p2.score += 1; results.append(f"{p1.name} 0 - 1 {p2.name}")
        elif val == "0.5":
            p1.score += 0.5; p2.score += 0.5; results.append(f"{p1.name} 0.5 - 0.5 {p2.name}")
        else:
            return f"Error en el formato del resultado para la mesa con {p1.name} vs {p2.name}. Usa '1', '0' o '0.5'."

    results_all.append(results)
    standings_text = standings_ui()
    crear_cartel_ronda(round_number, results=results, bye=bye_player, standings_text=standings_text)
    return "\n".join(results)

def standings_ui():
    global players, results_all
    if not players: return "No hay jugadores."
    clas = final_standings(players, results_all)
    output_lines = []
    for i,p in enumerate(clas):
        sb = calculate_sonneborn(p, players, results_all)
        buchholz = calculate_buchholz(p, players)
        progressive = calculate_progressive(p, results_all)
        output_lines.append(f"{i+1}. {p.name} - {p.score} pts | SB: {sb:.1f} | Buchholz: {buchholz:.1f} | Prog: {progressive:.1f}")
    return "\n".join(output_lines)

def download_csv_ui():
    global players, results_all
    if not players: return None
    return export_csv(players, results_all)

def list_pdfs_ui():
    files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith(".pdf")]
    return files if files else None

# ==== LANZAR INTERFAZ ====
with gr.Blocks() as demo:
    gr.Markdown("# ♟️ Gestor de Torneo Suizo (5 rondas, ganador único garantizado)")

    with gr.Tab("➕ Añadir jugadores"):
        txt_names = gr.Textbox(lines=10, label="Introduce nombres (uno por línea)")
        btn_add = gr.Button("Guardar")
        out_add = gr.Textbox(label="Estado")
        btn_add.click(add_players_ui, txt_names, out_add)

    with gr.Tab("🎲 Nueva Ronda"):
        btn_round = gr.Button("Generar emparejamientos")
        out_round = gr.Textbox(label="Emparejamientos")
        btn_round.click(new_round_ui, outputs=out_round)

    with gr.Tab("✍️ Guardar Resultados"):
        gr.Markdown("Introduce una línea por mesa: `1` gana el primero, `0` gana el segundo, `0.5` tablas")
        txt_results = gr.Textbox(lines=5, label="Resultados")
        btn_save = gr.Button("Guardar")
        out_save = gr.Textbox(label="Resultados guardados")
        btn_save.click(save_results_ui, txt_results, out_save)

    with gr.Tab("📊 Clasificación"):
        btn_stand = gr.Button("Ver standings")
        out_stand = gr.Textbox(label="Tabla")
        btn_stand.click(standings_ui, outputs=out_stand)

    with gr.Tab("⬇️ Descargar CSV"):
        btn_csv = gr.Button("Generar CSV")
        out_csv = gr.File()
        btn_csv.click(download_csv_ui, outputs=out_csv)

    with gr.Tab("📂 PDFs"):
        btn_pdfs = gr.Button("Listar PDFs")
        out_pdfs = gr.File(file_types=[".pdf"], file_count="multiple")
        btn_pdfs.click(list_pdfs_ui, outputs=out_pdfs)

if __name__ == "__main__":
    demo.launch()