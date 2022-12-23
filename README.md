# Project: Kitchenware Classification
Project created at cohort 2022 of ML Zoomcamp course.

The dataset I downloaded  from the kaggle.com competition at:
[Kitchenware Classification](https://www.kaggle.com/competitions/kitchenware-classification/data)


## Classification
From the competition I needed to classify images of different kitchenware items into 6 classes:

* cups
* glasses
* plates
* spoons
* forks
* knives

The files I downloaded to the following location

## Files

* data/train.csv - the training set (Image IDs and classes)
* data/test.csv - the test set (Just image IDs)
* data/sample_submission.csv - a sample submission file in the correct format (this I did not use)
* data/images/ - the images in the JPEG format

## Building the Model

For building the model I used the notebook keras-learning.ipynb. 
Where I used the training images.

I have build the model using keras tensorflow.

## Providing a service

* predict_service.py - python script with the flask service
* test_service.py - python script to test the service

I have created the predict_service.py which is taking a image url and
predicts which kind of kitchenware it is.

The script test_service.py is using a local file to test if the prediction is correct.




