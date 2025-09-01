#!/bin/bash
# Kafka Topic Creation Script for Market Scanning Engine
# Creates all topics with optimized configurations for high-throughput, low-latency processing

set -e

KAFKA_BOOTSTRAP_SERVERS=${KAFKA_BOOTSTRAP_SERVERS:-"localhost:9092"}
REPLICATION_FACTOR=${REPLICATION_FACTOR:-3}

echo "Creating Kafka topics for Market Scanning Engine..."
echo "Bootstrap servers: $KAFKA_BOOTSTRAP_SERVERS"
echo "Default replication factor: $REPLICATION_FACTOR"

# Raw Market Data Topics
echo "Creating raw market data topics..."
kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic market-data-raw \
  --partitions 24 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=86400000 \
  --config segment.ms=3600000 \
  --config compression.type=lz4 \
  --config min.insync.replicas=2 \
  --config max.message.bytes=1048576 \
  --config segment.bytes=1073741824 \
  --if-not-exists

kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic market-data-equity \
  --partitions 12 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=604800000 \
  --config segment.ms=3600000 \
  --config compression.type=lz4 \
  --config min.insync.replicas=2 \
  --config segment.bytes=536870912 \
  --if-not-exists

kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic market-data-options \
  --partitions 8 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=604800000 \
  --config compression.type=lz4 \
  --config min.insync.replicas=2 \
  --if-not-exists

kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic market-data-forex \
  --partitions 6 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=2592000000 \
  --config compression.type=lz4 \
  --if-not-exists

# News and Sentiment Topics
echo "Creating news and sentiment topics..."
kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic news-feed-raw \
  --partitions 8 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=2592000000 \
  --config segment.ms=7200000 \
  --config compression.type=gzip \
  --if-not-exists

kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic sentiment-analysis \
  --partitions 6 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=604800000 \
  --config compression.type=gzip \
  --if-not-exists

kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic social-sentiment \
  --partitions 4 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=259200000 \
  --config compression.type=gzip \
  --if-not-exists

# Economic Data Topics
echo "Creating economic data topics..."
kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic economic-indicators \
  --partitions 2 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=compact \
  --config retention.ms=7776000000 \
  --config compression.type=gzip \
  --config min.compaction.lag.ms=3600000 \
  --if-not-exists

# Signal Generation Topics
echo "Creating signal generation topics..."
kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic signals-generated \
  --partitions 16 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=2592000000 \
  --config segment.ms=1800000 \
  --config compression.type=lz4 \
  --config min.insync.replicas=2 \
  --config max.message.bytes=2097152 \
  --if-not-exists

kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic signals-validated \
  --partitions 12 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=2592000000 \
  --config compression.type=lz4 \
  --if-not-exists

# Risk Assessment Topics
echo "Creating risk assessment topics..."
kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic risk-scores \
  --partitions 8 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=2592000000 \
  --config compression.type=lz4 \
  --config min.insync.replicas=2 \
  --if-not-exists

kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic risk-alerts \
  --partitions 4 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=7776000000 \
  --config compression.type=gzip \
  --if-not-exists

# User Matching Topics
echo "Creating user matching topics..."
kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic user-signals-matched \
  --partitions 8 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=604800000 \
  --config compression.type=lz4 \
  --if-not-exists

# Notification Topics
echo "Creating notification topics..."
kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic notifications-queue \
  --partitions 6 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=259200000 \
  --config compression.type=gzip \
  --config min.insync.replicas=2 \
  --if-not-exists

kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic notifications-delivered \
  --partitions 4 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=604800000 \
  --config compression.type=gzip \
  --if-not-exists

kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic notifications-failed \
  --partitions 2 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=604800000 \
  --config compression.type=gzip \
  --if-not-exists

# Dead Letter Topics
echo "Creating dead letter topics..."
kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic dead-letter-queue \
  --partitions 4 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=2592000000 \
  --config compression.type=gzip \
  --if-not-exists

# Monitoring Topics
echo "Creating monitoring topics..."
kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic system-metrics \
  --partitions 4 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=604800000 \
  --config compression.type=gzip \
  --if-not-exists

kafka-topics --create --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic application-logs \
  --partitions 8 \
  --replication-factor $REPLICATION_FACTOR \
  --config cleanup.policy=delete \
  --config retention.ms=604800000 \
  --config compression.type=gzip \
  --if-not-exists

echo "All topics created successfully!"

# List all topics to verify
echo "Listing all topics:"
kafka-topics --list --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS

echo "Topic creation completed!"