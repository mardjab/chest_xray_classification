# Chest X-ray classification

## Overview

In the medical field, one of the most promising areas of innovation is the application of machine learning in medical imaging.

Chest X ray images currently are the most realiable method for diagnosing lung diseases.

In this project, a large set of chest X-ray images will be used to build a model that detects and classifies lungs infected with tuberculosis, pneumonia or COVID-19.

## Goals

To build a model that can be used to detect and classify human pulmonary diseases from chest x-ray images


## Dataset

https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis?select=train


## Milestones

In this project I was able to:

* Import a total of 6326 training images, 771 testing images and 38 validation images

* Augment images using the ImageDataGenerator class

* Build a model based on InceptionResNetV2 that classifies chest X-ray images into 4 classes (normal, tuberculosis, pneumonia or COVID-19) with 96.8% accuracy

