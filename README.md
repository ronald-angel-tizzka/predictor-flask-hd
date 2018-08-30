# Hand Written Digits Predictor Service

This project creates a service to classify hand written digits.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need to have docker installed in yur local machine. Example for ubuntu:

```
sudo apt install docker-ce
```

### Installing

Follow this instructions to install the environment.

* Download the project in you local machine.
```
git clone git clone https://ronald-velocity@bitbucket.org/ronald-velocity/predictor-flask-hd.git
```

**TODO:** This could be within the DockerFile.
**TODO:** In the future use Docker Compose to deploy multiple containers.

* Build your docker container using the Dockerfile.

```
docker build -t "hm-predictor:dockerfile" .
```

* Run your application. 

```
docker run -p 5000:5000  "hm-predictor:dockerfile".
```
## Running service calls

We can call the service endpoints on this way (using sample data):

* Training:

```
curl -X POST -H "Content-Type: application/json" -d '{"images_batch_path": "mnist.npz"}' http://localhost:5000/train_digits
```

sample Output: "True".

* Prediction:

```
curl -X POST -H "Content-Type: application/json" -d '{"image_path": "test/data/n5.p"}' http://localhost:5000/process_digit
```
sample Output: "[6]".


## Running the tests

The unit test is located in test.classification_service_test.py

to run add:
 
 ```
 python -m classification_service_test.py
```


## Architecture


## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

