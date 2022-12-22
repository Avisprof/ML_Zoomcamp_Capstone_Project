import numpy as np
import bentoml
from bentoml.io import Image, JSON
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.nn import softmax

runner = bentoml.keras.get("inception_v3_kitchen:mihd4xebh2a7lgsk").to_runner()
svc = bentoml.Service(name="kitchen_classification", runners=[runner])

classes = np.array(['cup', 'fork', 'glass', 'knife', 'plate', 'spoon'])

@svc.api(input=Image(), output=JSON())
async def classify(img):

    # preprocess image
    img = img.resize((299, 299))
    arr = np.array(img, dtype='float64')
    arr = np.array([arr])
    arr = preprocess_input(arr)

    # make prediction
    y_pred = await runner.async_run(arr)

    # get image class
    image_class = classes[y_pred.argmax(axis=1)][0]

    # get probability of image classes
    class_proba = softmax(y_pred[0]).numpy()*100
    class_proba_dict = {k: v for k, v in zip(classes, class_proba)}
    class_proba_dict = {k: f'{v:.1f}'  for k, v in sorted(class_proba_dict.items(), key=lambda item: item[1], reverse=True)}
    
    return {'image_class': image_class, 'class_proba': class_proba_dict}