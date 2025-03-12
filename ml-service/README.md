# Heart Disease Prediction Service

This repository contains the machine learning service for predicting heart disease using the OPENEMR dataset. The service is designed to integrate with the OPENEMR system to provide real-time predictions based on patient data.

## Features

- **Data Preprocessing**: Cleans and prepares the dataset for training.
- **Model Training**: Trains a machine learning model using the preprocessed data.
- **Prediction Service**: Provides an API for making predictions based on new patient data.
- **Integration**: Seamlessly integrates with the OPENEMR system.

## Getting Started

The code is created as a part of converting the OpenEMR system into a distributed system. The ML service can be added as a separate container to the Docker network after following below instructions. 

1. Create a directory called `prediction` inside the docker directory. 
2. Clone the repo into the `prediction` directory. The order should be as below.

```txt
root-directory
|__prediction
    |__ml-service
        |__Dockerfile  
        |__Train_Labels.csv  
        |__model.py  
        |__requirements.txt  
        |__train_values.csv  
        |__wait-for-kafka.sh
    |__ml_service.py
|__docker-compose-yml
|__kafka-init.sh
|__[OTHER FILES...]        
```

3. Add below code snippet to the `docker-compose.yml` file. 

```yml
zookeeper:
    image: confluentinc/cp-zookeeper:latest
    container_name: zookeeper
    restart: always
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    ports:
      - "2181:2181"

  kafka-setup:
    image: wurstmeister/kafka:2.13-2.8.1
    depends_on:
      - kafka-1
      - kafka-2
      - kafka-3
    command: ["/bin/bash", "-c", "./kafka-init.sh"]

  # Kafka Connect with Debezium
  kafka-1:
    image: wurstmeister/kafka:2.13-2.8.1
    container_name: kafka-1
    restart: always
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 2
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-1:9092
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
    ports:
      - "9092:9092"

  kafka-2:
    image: wurstmeister/kafka:2.13-2.8.1
    container_name: kafka-2
    restart: always
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 2
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-2:9093
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
    ports:
      - "9093:9093"

  kafka-3:
    image: wurstmeister/kafka:2.13-2.8.1
    container_name: kafka-3
    restart: always
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 3
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9094
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka-3:9094
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
    ports:
      - "9094:9094"

  kafka-connect:
    image: debezium/connect:2.4
    container_name: kafka-connect
    restart: always
    depends_on:
      - kafka-1
      - kafka-2
      - kafka-3
    environment:
      BOOTSTRAP_SERVERS: kafka-1:9092,kafka-2:9093,kafka-3:9094
      GROUP_ID: "1"
      CONFIG_STORAGE_TOPIC: debezium_configs
      OFFSET_STORAGE_TOPIC: debezium_offsets
      STATUS_STORAGE_TOPIC: debezium_status
      KEY_CONVERTER_SCHEMAS_ENABLE: "false"
      VALUE_CONVERTER_SCHEMAS_ENABLE: "false"
    ports:
      - "8083:8083"
  mlservice:
    restart: always
    build: ./prediction/ml-service
    ports:
    - 5001:5001
    volumes:
    - mlservicevolume:/model
    depends_on:
    - openemr
    - mysql
    - kafka-1
    - kafka-2
    - kafka-3
    - mongodb
    environment:
      KAFKA_BROKER: "kafka-1:9092,kafka-2:9093,kafka-3:9094"
  mongodb:
    image: mongo:6.0
    container_name: mongodb
    restart: always
    ports:
      - 27017:27017
    volumes:
      - mongodbvolume:/data/db
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password
```

Change the names of `kafka` nodes as needed. 

4. Add `patient_data_updates` to the `topic.prefix` in your `debezium-mysql.json`. 