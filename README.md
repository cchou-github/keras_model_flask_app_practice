# Overview
This is a Flask application that classifies the breed of the dog in the image.

# About the Machine Learing Model
The model this application used is built under the instruction of the machine learning course in Udemy:
https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/

And it is trained to classify dog breeds by using transfer learning from a pre-trained model mobilenetV2 .

You can check the whole steps in detail of building this model in my Google Colab:
https://colab.research.google.com/drive/1n8nhzbG5V5Vf-tmiliWzSVVhA4gI3Om4?usp=sharing

# How to Start This Web Application
1. Make sure you have docker and docker-compose in you machine and run the command below to start the local web server in the docker container.
```
docker-compose up
```

2. Access the URL via web browser.
```
http://localhost:5000/
```

## The Tools and Libraries Used
### For the model:
- Google Colab
- Python
- Tensorflow
- Tensorflow Hub
- Keras
- Numpy
- Pandas
- Metplotlib
### For the web:
- Python
- Flask
- Docker
- Numpy
- Metplotlib