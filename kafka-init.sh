#!/bin/bash

# Wait for Kafka to be ready
sleep 10

# Create topic for patient data updates
kafka-topics.sh --create --topic patient_data_updates --bootstrap-server kafka-1:9092 --replication-factor 1 --partitions 1

echo "Kafka topic 'patient_data_updates' created!"
