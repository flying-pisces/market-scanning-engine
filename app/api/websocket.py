"""
WebSocket API for real-time notifications
Provides live updates for signals, matches, and market data
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timezone
from fastapi import WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

from app.core.kafka_client import KafkaConsumerClient, KafkaTopics, kafka_config
from app.models.user import User

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and message broadcasting"""
    
    def __init__(self):
        # Active connections organized by user and connection type
        self.connections: Dict[str, Dict[str, Set[WebSocket]]] = {}
        self.user_connections: Dict[int, Set[WebSocket]] = {}  # user_id -> connections
        
        # Background consumers for Kafka topics
        self.consumers: List[KafkaConsumerClient] = []
        self.is_running = False
    
    async def connect(self, websocket: WebSocket, user_id: int, connection_type: str = "general"):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        # Initialize user connection tracking
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        
        if connection_type not in self.connections:
            self.connections[connection_type] = {}
        
        if str(user_id) not in self.connections[connection_type]:
            self.connections[connection_type][str(user_id)] = set()
        
        # Add connection
        self.user_connections[user_id].add(websocket)
        self.connections[connection_type][str(user_id)].add(websocket)
        
        logger.info(f"WebSocket connected: user_id={user_id}, type={connection_type}")
        
        # Send welcome message
        await self.send_to_connection(websocket, {
            "type": "connection_established",
            "user_id": user_id,
            "connection_type": connection_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def disconnect(self, websocket: WebSocket, user_id: int, connection_type: str = "general"):
        """Handle WebSocket disconnection"""
        try:
            # Remove from user connections
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(websocket)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            # Remove from typed connections
            if (connection_type in self.connections and 
                str(user_id) in self.connections[connection_type]):
                self.connections[connection_type][str(user_id)].discard(websocket)
                if not self.connections[connection_type][str(user_id)]:
                    del self.connections[connection_type][str(user_id)]
            
            logger.info(f"WebSocket disconnected: user_id={user_id}, type={connection_type}")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def send_to_user(self, user_id: int, message: Dict[str, Any]):
        """Send message to all connections for a specific user"""
        if user_id not in self.user_connections:
            return
        
        disconnected = set()
        for websocket in self.user_connections[user_id].copy():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to user {user_id}: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for ws in disconnected:
            self.user_connections[user_id].discard(ws)
    
    async def send_to_connection_type(self, connection_type: str, message: Dict[str, Any]):
        """Send message to all connections of a specific type"""
        if connection_type not in self.connections:
            return
        
        for user_id, websockets in self.connections[connection_type].items():
            disconnected = set()
            for websocket in websockets.copy():
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send to {connection_type} connection for user {user_id}: {e}")
                    disconnected.add(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                websockets.discard(ws)
    
    async def send_to_connection(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to specific connection"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send message to connection: {e}")
    
    async def broadcast(self, message: Dict[str, Any], exclude_user: Optional[int] = None):
        """Broadcast message to all connected users"""
        for user_id, websockets in self.user_connections.items():
            if exclude_user and user_id == exclude_user:
                continue
            
            disconnected = set()
            for websocket in websockets.copy():
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to broadcast to user {user_id}: {e}")
                    disconnected.add(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                websockets.discard(ws)
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return sum(len(connections) for connections in self.user_connections.values())
    
    def get_user_connection_count(self, user_id: int) -> int:
        """Get number of connections for specific user"""
        return len(self.user_connections.get(user_id, set()))
    
    async def start_kafka_consumers(self):
        """Start Kafka consumers for real-time message forwarding"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Set up consumers for different message types
        consumers_config = [
            {
                "topics": [KafkaTopics.SIGNALS_GENERATED],
                "group_id": "websocket-signals",
                "handler": self._handle_signal_message
            },
            {
                "topics": [KafkaTopics.SIGNALS_MATCHED],
                "group_id": "websocket-matches",
                "handler": self._handle_match_message
            },
            {
                "topics": [KafkaTopics.MARKET_DATA_RAW],
                "group_id": "websocket-market-data",
                "handler": self._handle_market_data_message
            },
            {
                "topics": [KafkaTopics.NOTIFICATIONS],
                "group_id": "websocket-notifications", 
                "handler": self._handle_notification_message
            },
            {
                "topics": [KafkaTopics.SYSTEM_METRICS],
                "group_id": "websocket-system",
                "handler": self._handle_system_message
            }
        ]
        
        # Start consumer tasks
        tasks = []
        for config in consumers_config:
            consumer = KafkaConsumerClient(
                config=kafka_config,
                group_id=config["group_id"],
                topics=config["topics"]
            )
            self.consumers.append(consumer)
            
            task = asyncio.create_task(
                self._consume_messages(consumer, config["handler"])
            )
            tasks.append(task)
        
        logger.info(f"Started {len(tasks)} WebSocket Kafka consumers")
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in Kafka consumer tasks: {e}")
        finally:
            await self._stop_consumers()
    
    async def _stop_consumers(self):
        """Stop all Kafka consumers"""
        self.is_running = False
        for consumer in self.consumers:
            consumer.close()
        self.consumers.clear()
        logger.info("Stopped WebSocket Kafka consumers")
    
    async def _consume_messages(self, consumer: KafkaConsumerClient, handler):
        """Consume messages from Kafka and forward via WebSocket"""
        async for message_info in consumer.consume_messages(handler):
            if not self.is_running:
                break
    
    async def _handle_signal_message(self, topic: str, message: Dict[str, Any]):
        """Handle new trading signals"""
        try:
            websocket_message = {
                "type": "signal_generated",
                "data": {
                    "symbol": message.get("symbol"),
                    "asset_class": message.get("asset_class"),
                    "signal_type": message.get("signal_type"),
                    "risk_score": message.get("risk_score"),
                    "confidence": message.get("confidence"),
                    "entry_price": message.get("entry_price"),
                    "target_price": message.get("target_price"),
                    "description": message.get("description"),
                    "timeframe": message.get("timeframe")
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Broadcast to all users with signal subscriptions
            await self.send_to_connection_type("signals", websocket_message)
            
        except Exception as e:
            logger.error(f"Error handling signal message: {e}")
    
    async def _handle_match_message(self, topic: str, message: Dict[str, Any]):
        """Handle signal-user matches"""
        try:
            # Extract user matches if present
            matches = message.get("matches", [])
            signal_data = message.get("signal", {})
            
            for match in matches:
                user_id = match.get("user_id")
                if not user_id:
                    continue
                
                websocket_message = {
                    "type": "signal_match",
                    "data": {
                        "signal": {
                            "symbol": signal_data.get("symbol"),
                            "signal_type": signal_data.get("signal_type"),
                            "risk_score": signal_data.get("risk_score"),
                            "entry_price": signal_data.get("entry_price"),
                            "description": signal_data.get("description")
                        },
                        "match_score": match.get("match_score"),
                        "compatibility_score": match.get("compatibility_score"),
                        "reason": match.get("reason", "Signal matches your risk profile")
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Send to specific user
                await self.send_to_user(user_id, websocket_message)
            
        except Exception as e:
            logger.error(f"Error handling match message: {e}")
    
    async def _handle_market_data_message(self, topic: str, message: Dict[str, Any]):
        """Handle market data updates"""
        try:
            # Only forward significant price movements to avoid spam
            price_change = message.get("price_change_pct")
            if price_change and abs(price_change) >= 1.0:  # 1% or more movement
                
                websocket_message = {
                    "type": "market_update",
                    "data": {
                        "symbol": message.get("symbol"),
                        "price": message.get("price"),
                        "price_change": price_change,
                        "volume": message.get("volume"),
                        "timestamp": message.get("timestamp")
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Broadcast to market data subscribers
                await self.send_to_connection_type("market_data", websocket_message)
            
        except Exception as e:
            logger.error(f"Error handling market data message: {e}")
    
    async def _handle_notification_message(self, topic: str, message: Dict[str, Any]):
        """Handle user notifications"""
        try:
            user_id = message.get("user_id")
            if not user_id:
                return
            
            websocket_message = {
                "type": "notification",
                "data": {
                    "title": message.get("title"),
                    "message": message.get("message"),
                    "priority": message.get("priority", "normal"),
                    "category": message.get("category", "general")
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Send to specific user
            await self.send_to_user(user_id, websocket_message)
            
        except Exception as e:
            logger.error(f"Error handling notification message: {e}")
    
    async def _handle_system_message(self, topic: str, message: Dict[str, Any]):
        """Handle system status messages"""
        try:
            # Forward system status to all connected users
            if message.get("type") == "market_hours":
                websocket_message = {
                    "type": "market_status",
                    "data": {
                        "is_open": message.get("is_open"),
                        "session": message.get("session"),
                        "next_open": message.get("next_open"),
                        "next_close": message.get("next_close")
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                await self.broadcast(websocket_message)
            
        except Exception as e:
            logger.error(f"Error handling system message: {e}")


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


@asynccontextmanager
async def websocket_lifespan():
    """Context manager for WebSocket service lifecycle"""
    try:
        # Start Kafka consumers in background
        consumer_task = asyncio.create_task(websocket_manager.start_kafka_consumers())
        yield websocket_manager
    finally:
        # Clean shutdown
        websocket_manager.is_running = False
        if 'consumer_task' in locals():
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass
        await websocket_manager._stop_consumers()


async def websocket_endpoint(websocket: WebSocket, user_id: int, connection_type: str = "general"):
    """WebSocket endpoint handler"""
    await websocket_manager.connect(websocket, user_id, connection_type)
    
    try:
        while True:
            # Listen for client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle client commands
            await handle_client_message(websocket, user_id, message)
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, user_id, connection_type)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        websocket_manager.disconnect(websocket, user_id, connection_type)


async def handle_client_message(websocket: WebSocket, user_id: int, message: Dict[str, Any]):
    """Handle messages from WebSocket clients"""
    try:
        message_type = message.get("type")
        
        if message_type == "subscribe":
            # Handle subscription requests
            subscription_type = message.get("subscription")
            if subscription_type in ["signals", "market_data", "notifications"]:
                # Move connection to appropriate group
                if subscription_type not in websocket_manager.connections:
                    websocket_manager.connections[subscription_type] = {}
                if str(user_id) not in websocket_manager.connections[subscription_type]:
                    websocket_manager.connections[subscription_type][str(user_id)] = set()
                
                websocket_manager.connections[subscription_type][str(user_id)].add(websocket)
                
                await websocket_manager.send_to_connection(websocket, {
                    "type": "subscription_confirmed",
                    "subscription": subscription_type,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        elif message_type == "unsubscribe":
            # Handle unsubscription requests
            subscription_type = message.get("subscription")
            if (subscription_type in websocket_manager.connections and
                str(user_id) in websocket_manager.connections[subscription_type]):
                websocket_manager.connections[subscription_type][str(user_id)].discard(websocket)
        
        elif message_type == "ping":
            # Handle keepalive pings
            await websocket_manager.send_to_connection(websocket, {
                "type": "pong",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        else:
            logger.warning(f"Unknown message type from user {user_id}: {message_type}")
    
    except Exception as e:
        logger.error(f"Error handling client message: {e}")


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance"""
    return websocket_manager