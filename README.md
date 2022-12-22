# Kitchenware image classification ([Kaggle Competition](https://www.kaggle.com/competitions/kitchenware-classification)) 

In this competition we need to classify images of different kitchenware items into 6 classes:

* cups
* glasses
* plates
* spoons
* forks
* knives

You can get this dataset from [kaggle](https://www.kaggle.com/competitions/kitchenware-classification)

## About files and folders in this repo

|  File name |      Description       |
|:--------:|:-----------------------------------:|
|    README.md   |Details about this project| 
|    notebook.ipynb   |Jupyter notebook file with EDA, training different models and choose the best model |
|    train.py   |Python script for training InceptionV3 model |
|    kitchenware_image_classiffication.h5   |Saved fitted InvceptionV3 model |
|    predict.ipynb   |Jupyter notebook file that loads the InvceptionV3 and make predictions|
|    Pipfile   |The concrete requirements for a Python Application|
|    Pipfile.lock   |The details of the environment|
|    bentofile.yaml   |Text document that contains all the commands to create docker container image|
|    bentoml_service.py   |Script to run bentoml model |
|    using_bentoml_model.ipynb   |Jupyter notebook file with example to predict image class based on bentoml deploy model|
 
## Dataset information:
This dataset consist of 5559 labeled images. The information about classes is in the train.py

## How to reproduce the project
Clone this repo in your local machine with the command:
```
git clone https://github.com/Avisprof/ML_Zoomcamp_Midterm_Project.git
```
If you haven't installed `pipenv` yet, you need to do it with:
```
pip install pipenv
```
Then you can recreate the environment by running the below command in the project directory:
```
pipenv --python 3.9
pipenv install .
```

## Deploy Bentoml model
Run train.py script to fit the model and save model to the file
```
python train.py
```

Prepare yaml file:
```
service: "bentoml_service.py:svc"
labels:
  owner: avisprov
  project: kitchenware_classification
include:
  - "bentoml_service.py"
python:
  packages:
    - tensorflow
    - Pillow  
```

Build model with command:
```
bentoml build
```

Build docker image:
```
bentoml containerize kitchen_classification:qwjcdleb4wibvx7k
```

Run docker image 
```
docker run -it --rm -p 3000:3000 kitchen_classification:qwjcdleb4wibvx7k serve --production
```

To check model you can run jupyter notebook file:
[predict.ipynb](predict.ipynb)

Then open http://localhost:3000 in your browser and load image to prediction

![bento_choose_file](pic/bento_choose_file.jpg)

Choose the image file and press the "Execute" button

![pic/bento_predictions](pic/bento_predictions.jpg)


## Using docker image for deploy

You can donwload docker image:
```
docker push interests/kitchenware_image_classification:latest
```

And than run this image to start the BentoServer:
```
docker run -it -p 3000:3000 interests/kitchenware_image_classification:latest serve --production
```

Then open http://localhost:3000 in your browser and load image to prediction

## Cloud deploy

To check cloud model you can run jupyter notebook file:
[using_bentoml_model.ipynb](using_bentoml_model.ipynb)

Ore you can make use http://79.137.198.248:3000/ with cloud deployment, press "Post" button, after that press "Try it out" button, choose the image file and press the "Execute" button
