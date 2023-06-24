from MLcommon.loader import load_zip_model
import cv2 as cv

#Cargar modelo
model = load_zip_model('assets/coco_detector.ml', 'assets/architecture.zip')

# Realizar predicciones con los frames
im = cv.imread('assets/im.png')
predictions = model.predict(im)

print(predictions)
# Predictions = [Object('subobject': None, 'geometry': BoundBox([248, 221, 261, 269]), 'label': 'tie', 'score': 0.6527128219604492), Object('subobject': None, 'geometry': BoundBox([70, 214, 110, 323]), 'label': 'handbag', 'score': 0.443928986787796), Object('subobject': None, 'geometry': BoundBox([69, 268, 104, 322]), 'label': 'handbag', 'score': 0.7130095362663269), Object('subobject': None, 'geometry': BoundBox([379, 137, 386, 155]), 'label': 'traffic light', 'score': 0.3744637668132782), Object('subobject': None, 'geometry': BoundBox([381, 139, 389, 156]), 'label': 'traffic light', 'score': 0.5535294413566589), Object('subobject': None, 'geometry': BoundBox([371, 137, 378, 153]), 'label': 'traffic light', 'score': 0.6306472420692444), Object('subobject': None, 'geometry': BoundBox([404, 231, 416, 271]), 'label': 'car', 'score': 0.4761880338191986), Object('subobject': None, 'geometry': BoundBox([97, 203, 326, 335]), 'label': 'car', 'score': 0.7539923787117004), Object('subobject': None, 'geometry': BoundBox([10, 216, 101, 302]), 'label': 'car', 'score': 0.8587526679039001), Object('subobject': None, 'geometry': BoundBox([325, 196, 410, 284]), 'label': 'car', 'score': 0.9630563259124756), Object('subobject': None, 'geometry': BoundBox([220, 189, 280, 397]), 'label': 'person', 'score': 0.9941640496253967), Object('subobject': None, 'geometry': BoundBox([86, 195, 146, 399]), 'label': 'person', 'score': 0.9968808889389038)]
