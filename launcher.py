import webbrowser
import threading
import time
import main  # Importa tu app con demo.launch()

def run_app():
    # Arranca la app de Gradio
    main.demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, inbrowser=False)

if __name__ == "__main__":
    # Inicia el servidor en un hilo aparte
    t = threading.Thread(target=run_app)
    t.daemon = True
    t.start()

    # Espera un poco a que Gradio arranque
    time.sleep(2)

    # Abre navegador automáticamente
    webbrowser.open("http://127.0.0.1:7860")

    # Mantén el programa vivo
    t.join()
