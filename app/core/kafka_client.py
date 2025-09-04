"""
Kafka client for real-time message streaming
Handles producers and consumers for market data pipeline
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass

from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
import aioredis

logger = logging.getLogger(__name__)


@dataclass
class KafkaConfig:
    """Kafka configuration"""
    bootstrap_servers: str = "localhost:9092"
    client_id: str = "market-scanner"
    acks: str = "1"  # Wait for leader acknowledgment
    retries: int = 3
    batch_size: int = 16384
    linger_ms: int = 10
    buffer_memory: int = 33554432
    compression_type: str = "snappy"


class KafkaTopics:
    """Kafka topic definitions"""
    
    # Market data topics
    MARKET_DATA_RAW = "market-data-raw"
    MARKET_DATA_VALIDATED = "market-data-validated"
    TECHNICAL_INDICATORS = "technical-indicators"
    OPTIONS_DATA = "options-data"
    
    # Signal topics
    SIGNALS_GENERATED = "signals-generated"
    SIGNALS_MATCHED = "signals-matched"
    SIGNALS_EXPIRED = "signals-expired"
    
    # User interaction topics
    USER_ACTIONS = "user-actions"
    NOTIFICATIONS = "notifications"
    
    # System topics
    SYSTEM_METRICS = "system-metrics"
    ERROR_EVENTS = "error-events"
    
    @classmethod
    def get_all_topics(cls) -> List[str]:
        """Get all defined topics"""
        return [
            cls.MARKET_DATA_RAW,
            cls.MARKET_DATA_VALIDATED,
            cls.TECHNICAL_INDICATORS,
            cls.OPTIONS_DATA,
            cls.SIGNALS_GENERATED,
            cls.SIGNALS_MATCHED,
            cls.SIGNALS_EXPIRED,
            cls.USER_ACTIONS,
            cls.NOTIFICATIONS,
            cls.SYSTEM_METRICS,
            cls.ERROR_EVENTS,
        ]


class KafkaProducerClient:
    """Async Kafka producer wrapper"""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producer = None
        self._setup_producer()
    
    def _setup_producer(self):
        """Initialize Kafka producer"""
        producer_config = {
            'bootstrap.servers': self.config.bootstrap_servers,
            'client.id': f"{self.config.client_id}-producer",
            'acks': self.config.acks,
            'retries': self.config.retries,
            'batch.size': self.config.batch_size,
            'linger.ms': self.config.linger_ms,
            'buffer.memory': self.config.buffer_memory,
            'compression.type': self.config.compression_type,
            'enable.idempotence': True,  # Exactly-once semantics
        }
        
        self.producer = Producer(producer_config)
        logger.info("Kafka producer initialized")
    
    async def send_message(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """Send a message to Kafka topic"""
        
        try:
            # Add metadata
            message_with_metadata = {
                **message,
                "timestamp": datetime.utcnow().isoformat(),
                "source": self.config.client_id,
            }
            
            # Serialize message
            serialized_message = json.dumps(message_with_metadata).encode('utf-8')
            serialized_key = key.encode('utf-8') if key else None
            
            # Convert headers to bytes
            kafka_headers = None
            if headers:
                kafka_headers = [(k, v.encode('utf-8')) for k, v in headers.items()]
            
            # Send message
            self.producer.produce(
                topic=topic,
                key=serialized_key,
                value=serialized_message,
                headers=kafka_headers,
                callback=self._delivery_callback
            )
            
            # Trigger delivery report callbacks
            self.producer.poll(0)
            
            return True
            
        except KafkaException as e:
            logger.error(f"Failed to send message to {topic}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message to {topic}: {e}")
            return False
    
    def _delivery_callback(self, err, msg):
        """Callback for message delivery confirmation"""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    async def send_batch(
        self,
        topic: str,
        messages: List[Dict[str, Any]],
        keys: Optional[List[str]] = None
    ) -> int:
        """Send multiple messages in batch"""
        
        success_count = 0
        
        for i, message in enumerate(messages):
            key = keys[i] if keys and i < len(keys) else None
            if await self.send_message(topic, message, key):
                success_count += 1
        
        # Flush to ensure delivery
        self.producer.flush(timeout=30)
        
        logger.info(f"Sent {success_count}/{len(messages)} messages to {topic}")
        return success_count
    
    def close(self):
        """Close producer and flush pending messages"""
        if self.producer:
            self.producer.flush(timeout=30)
            logger.info("Kafka producer closed")


class KafkaConsumerClient:
    """Async Kafka consumer wrapper"""
    
    def __init__(
        self,
        config: KafkaConfig,
        group_id: str,
        topics: List[str],
        auto_offset_reset: str = "earliest"
    ):
        self.config = config
        self.group_id = group_id
        self.topics = topics
        self.consumer = None
        self._setup_consumer(auto_offset_reset)
    
    def _setup_consumer(self, auto_offset_reset: str):
        """Initialize Kafka consumer"""
        consumer_config = {
            'bootstrap.servers': self.config.bootstrap_servers,
            'client.id': f"{self.config.client_id}-consumer",
            'group.id': self.group_id,
            'auto.offset.reset': auto_offset_reset,
            'enable.auto.commit': False,  # Manual commit for reliability
            'max.poll.interval.ms': 300000,  # 5 minutes
            'session.timeout.ms': 30000,
            'heartbeat.interval.ms': 10000,
            'fetch.min.bytes': 1024,
            'fetch.max.wait.ms': 500,
        }
        
        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe(self.topics)
        logger.info(f"Kafka consumer initialized for topics: {self.topics}")
    
    async def consume_messages(
        self,
        message_handler: Callable[[str, Dict[str, Any]], None],
        max_messages: int = 1000,
        timeout: float = 1.0
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Consume messages from subscribed topics"""
        
        processed_count = 0
        
        try:
            while processed_count < max_messages:
                # Poll for messages
                msg = self.consumer.poll(timeout=timeout)
                
                if msg is None:
                    await asyncio.sleep(0.1)  # Brief pause when no messages
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        break
                
                try:
                    # Deserialize message
                    message_data = json.loads(msg.value().decode('utf-8'))
                    
                    # Process message
                    await message_handler(msg.topic(), message_data)
                    
                    # Commit offset
                    self.consumer.commit(message=msg)
                    
                    processed_count += 1
                    
                    yield {
                        "topic": msg.topic(),
                        "partition": msg.partition(),
                        "offset": msg.offset(),
                        "key": msg.key().decode('utf-8') if msg.key() else None,
                        "data": message_data,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to deserialize message: {e}")
                    # Still commit to avoid reprocessing
                    self.consumer.commit(message=msg)
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Don't commit on processing error - will retry
                    break
                    
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            logger.info(f"Processed {processed_count} messages")
    
    def close(self):
        """Close consumer"""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")


class KafkaAdmin:
    """Kafka administration utilities"""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.admin_client = AdminClient({
            'bootstrap.servers': config.bootstrap_servers,
            'client.id': f"{config.client_id}-admin"
        })
    
    async def create_topics(
        self,
        topics: List[str],
        num_partitions: int = 6,
        replication_factor: int = 1
    ) -> bool:
        """Create Kafka topics if they don't exist"""
        
        try:
            # Get existing topics
            metadata = self.admin_client.list_topics(timeout=10)
            existing_topics = set(metadata.topics.keys())
            
            # Filter topics that don't exist
            topics_to_create = [
                topic for topic in topics
                if topic not in existing_topics
            ]
            
            if not topics_to_create:
                logger.info("All topics already exist")
                return True
            
            # Create new topics
            new_topics = [
                NewTopic(
                    topic=topic,
                    num_partitions=num_partitions,
                    replication_factor=replication_factor,
                    config={
                        'cleanup.policy': 'delete',
                        'retention.ms': '604800000',  # 7 days
                        'compression.type': 'snappy',
                    }
                )
                for topic in topics_to_create
            ]
            
            # Create topics
            future_results = self.admin_client.create_topics(
                new_topics,
                validate_only=False,
                request_timeout=30
            )
            
            # Wait for results
            success_count = 0
            for topic, future in future_results.items():
                try:
                    future.result()  # Block until completion
                    logger.info(f"Topic '{topic}' created successfully")
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to create topic '{topic}': {e}")
            
            logger.info(f"Created {success_count}/{len(topics_to_create)} topics")
            return success_count == len(topics_to_create)
            
        except Exception as e:
            logger.error(f"Topic creation failed: {e}")
            return False
    
    async def ensure_topics_exist(self) -> bool:
        """Ensure all required topics exist"""
        return await self.create_topics(KafkaTopics.get_all_topics())


# Global Kafka clients (initialized on startup)
kafka_config = KafkaConfig()
producer_client: Optional[KafkaProducerClient] = None
admin_client: Optional[KafkaAdmin] = None


async def init_kafka(bootstrap_servers: str = "localhost:9092") -> bool:
    """Initialize Kafka infrastructure"""
    global kafka_config, producer_client, admin_client
    
    try:
        # Update config
        kafka_config.bootstrap_servers = bootstrap_servers
        
        # Initialize clients
        producer_client = KafkaProducerClient(kafka_config)
        admin_client = KafkaAdmin(kafka_config)
        
        # Create required topics
        if await admin_client.ensure_topics_exist():
            logger.info("Kafka infrastructure initialized successfully")
            return True
        else:
            logger.error("Failed to create required topics")
            return False
            
    except Exception as e:
        logger.error(f"Kafka initialization failed: {e}")
        return False


async def close_kafka():
    """Close Kafka connections"""
    global producer_client
    
    if producer_client:
        producer_client.close()
        producer_client = None
    
    logger.info("Kafka connections closed")


def get_producer() -> Optional[KafkaProducerClient]:
    """Get the global Kafka producer"""
    return producer_client