 # Fedkso

This repo consists of a [FEDn](https://github.com/scaleoutsystems/fedn) client code for [KSO](https://github.com/ocean-data-factory-sweden/koster_yolov4). To set up FEDn which includes a reducer, combiner, and database-related base services, clone the FEDn [FEDn](https://github.com/scaleoutsystems/fedn) repository and follow the instructions in its [readme](https://github.com/scaleoutsystems/fedn#pseudo-distributed-deployment). The code has been tested with V0.3.1 tag of FEDn using `quay.io/minio/minio:latest` for the minio base service. 

# Introduction

Fedkso is a POC developed as part of the federated learning project funded by EOSC-Nordic. The POC involves training an object detection model in a federated learning setting using the baltic subsea movies dataset collected [Koster Seafloor Observatory](https://github.com/ocean-data-factory-sweden/koster_data_management). More details about the dataset and experiment setup are in the repository.

# Dataset

The test dataset used for test and development consists of 3 hours of subsea video footage.

# Annotations

The annotations provide the bounding boxes of the sea life present on the movies classifying them into species. Quality of the dataset: testing, small, real-world, challenging, single class, class-imbalanced dataset.

# Object Detection Model

For this POC we have used YOLOv5−tiny as our baseline model for object detection. The choice was dictated by considering conditions such as the architecture of the model. The architecture supports a relatively small available dataset, small number of classes, and facilitates fast training speeds on limited computational resources during test and development.

# Framework

We use an open-source framework called FEDn to communicate and coordinate between the local model training clients and the central model averaging server. The FEDn architecture can be seen from [here](https://scaleoutsystems.github.io/fedn/architecture.html). 


# Client Deployment (pseudo-distributed)

The easiest way to start clients for quick testing is by using Docker. We provide a docker-compose template for convenience.

```
docker-compose -f docker-compose.yaml -f private-network.yaml up
```

