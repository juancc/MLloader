# ML Loader
Ejemplo para predecir usando los modelos de Vaico
Para usar los modelos se necesitan dos archivos:
- Modelo con extensión .ml
- Arquitectura comprimida .zip

# Requerimientos
Los requerimientos se encuentran en el archivo requiremens.txt y se pueden instalar usando el comando
```bash
pip install -r requirements.txt
```
Estos requerimientos puden variar dependiendo del modelo que se use

Se sugiere usar un ambiente virtual (https://virtualenv.pypa.io/en/stable/) con python > 3.5

# Assets
En la carpeta assets se encuentran los archivos necesarios para predecir 
## COCO detector
Modelo con extensión .ml para predecir 90 clases con sus respectivo recuadro. El output de este modelo es una lista de objetos  clase que contiene la geometría, la etiqueta y el puntaje
La geometría que retorna es tipo boundbox definida por las coordenadas (xmin,xmax,ymin,ymax).
Ejemplo de retorno de predicción
```python
[Object('subobject': None, 'geometry': BoundBox([248, 221, 261, 269]), 'label': 'tie', 'score': 0.6527128219604492),]
```



## Architecture
Clase de la arquitectura del modelo
