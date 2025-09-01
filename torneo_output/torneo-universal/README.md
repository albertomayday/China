# Torneo Universal

Este proyecto es un gestor universal de torneos suizos, diseñado para soportar tanto competiciones individuales como por equipos. Permite la configuración de múltiples sistemas de puntuación y genera clasificaciones en formatos CSV y PDF.

## Estructura del Proyecto

- `src/Multijuego.py`: Contiene la lógica principal del gestor de torneos, incluyendo la definición de las clases `Player`, `Team`, `Game`, y `Tournament`. También incluye métodos para gestionar el estado del torneo, emparejamientos, resultados, y exportación de clasificaciones.

- `requirements.txt`: Lista las dependencias necesarias para el proyecto, incluyendo `gradio` para la interfaz de usuario y `reportlab` para la generación de PDFs.

## Instalación

Para instalar las dependencias del proyecto, asegúrate de tener `pip` instalado y ejecuta el siguiente comando en la raíz del proyecto:

```
pip install -r requirements.txt
```

## Ejecución

Para ejecutar el programa, utiliza el siguiente comando:

```
python src/Multijuego.py
```

## Uso

Una vez que el programa esté en ejecución, se abrirá una interfaz de usuario donde podrás:

- Configurar el modo de torneo (Individual o Equipos).
- Añadir jugadores y equipos.
- Generar rondas y emparejamientos.
- Introducir resultados y generar clasificaciones.
- Exportar clasificaciones en formato CSV y PDF.

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir, por favor abre un issue o un pull request en el repositorio.