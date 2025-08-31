# Proyecto Multijuego Torneo

Este proyecto es un gestor universal de torneos suizos, diseñado para soportar múltiples sistemas de puntuación configurables por juego. Permite la gestión de torneos tanto individuales como por equipos, y proporciona una interfaz de usuario intuitiva utilizando Gradio. Además, genera clasificaciones en formato PDF utilizando la biblioteca ReportLab.

## Estructura del Proyecto

- **src/Multijuego.py**: Contiene la implementación del gestor de torneos, incluyendo las clases `Player`, `Team`, `Game`, y `Tournament`. Maneja la lógica del torneo, emparejamientos, resultados y exportación de clasificaciones.
  
- **requirements.txt**: Lista las dependencias necesarias para el proyecto, incluyendo bibliotecas como `gradio` y `reportlab`.

## Instalación

Para instalar las dependencias del proyecto, asegúrate de tener `pip` instalado y ejecuta el siguiente comando en la raíz del proyecto:

```
pip install -r requirements.txt
```

## Uso

Para ejecutar el gestor de torneos, simplemente corre el archivo `Multijuego.py` en el directorio `src`. Esto iniciará la interfaz de usuario donde podrás configurar el torneo, añadir jugadores o equipos, generar rondas y guardar resultados.

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir, por favor abre un issue o envía un pull request con tus cambios.

## Licencia

Este proyecto está bajo la Licencia MIT.