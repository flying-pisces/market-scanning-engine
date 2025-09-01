"""
Market Data Pipeline DAG
Orchestrates the complete market data processing pipeline from ingestion to notification delivery.
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.http_sensor import HttpSensor
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup

import structlog

logger = structlog.get_logger(__name__)

# DAG configuration
default_args = {
    'owner': 'market-scanning-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(hours=1),
}

# DAG definition
dag = DAG(
    'market_data_pipeline',
    default_args=default_args,
    description='Complete market data processing pipeline',
    schedule_interval='@once',  # Runs continuously via sensors
    catchup=False,
    max_active_runs=1,
    max_active_tasks=10,
    tags=['market-data', 'real-time', 'production']
)


def check_market_hours(**context):
    """Check if markets are open for trading."""
    from datetime import datetime
    import pytz
    
    # Check major market hours
    now_utc = datetime.now(pytz.UTC)
    
    # US Markets (NYSE/NASDAQ): 9:30 AM - 4:00 PM ET
    et_tz = pytz.timezone('US/Eastern')
    et_time = now_utc.astimezone(et_tz)
    us_open = et_time.hour >= 9 and (et_time.hour < 16 or (et_time.hour == 9 and et_time.minute >= 30))
    us_weekday = et_time.weekday() < 5  # Monday = 0, Friday = 4
    
    # European Markets: 8:00 AM - 4:30 PM CET
    cet_tz = pytz.timezone('CET')
    cet_time = now_utc.astimezone(cet_tz)
    eu_open = cet_time.hour >= 8 and (cet_time.hour < 16 or (cet_time.hour == 16 and cet_time.minute <= 30))
    eu_weekday = cet_time.weekday() < 5
    
    # Asian Markets: Various times
    jst_tz = pytz.timezone('Asia/Tokyo')
    jst_time = now_utc.astimezone(jst_tz)
    asia_open = jst_time.hour >= 9 and jst_time.hour < 15
    asia_weekday = jst_time.weekday() < 5
    
    market_status = {
        'us_open': us_open and us_weekday,
        'eu_open': eu_open and eu_weekday,
        'asia_open': asia_open and asia_weekday,
        'any_open': (us_open and us_weekday) or (eu_open and eu_weekday) or (asia_open and asia_weekday)
    }
    
    logger.info("Market status checked", status=market_status)
    return market_status


def validate_data_quality(**context):
    """Validate data quality metrics."""
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    
    # Connect to PostgreSQL
    postgres = PostgresHook(postgres_conn_id='market_data_postgres')
    
    # Quality checks
    quality_checks = [
        {
            'name': 'data_freshness',
            'query': """
                SELECT COUNT(*) as count
                FROM market_data.price_data
                WHERE created_at >= NOW() - INTERVAL '5 minutes'
            """,
            'expected_min': 1000  # At least 1000 records in last 5 minutes
        },
        {
            'name': 'duplicate_check',
            'query': """
                SELECT COUNT(*) as duplicates
                FROM (
                    SELECT symbol, timestamp, COUNT(*)
                    FROM market_data.price_data
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                    GROUP BY symbol, timestamp
                    HAVING COUNT(*) > 1
                ) t
            """,
            'expected_max': 10  # Less than 10 duplicates per hour
        },
        {
            'name': 'price_anomaly_check',
            'query': """
                SELECT COUNT(*) as anomalies
                FROM market_data.price_data
                WHERE created_at >= NOW() - INTERVAL '1 hour'
                AND (price <= 0 OR price > 1000000)
            """,
            'expected_max': 0  # No price anomalies
        }
    ]
    
    results = {}
    failed_checks = []
    
    for check in quality_checks:
        try:
            result = postgres.get_first(check['query'])
            count = result[0] if result else 0
            results[check['name']] = count
            
            # Validate against expectations
            if 'expected_min' in check and count < check['expected_min']:
                failed_checks.append(f"{check['name']}: {count} < {check['expected_min']}")
            if 'expected_max' in check and count > check['expected_max']:
                failed_checks.append(f"{check['name']}: {count} > {check['expected_max']}")
                
        except Exception as e:
            logger.error(f"Quality check failed: {check['name']}", error=str(e))
            failed_checks.append(f"{check['name']}: Error - {str(e)}")
    
    if failed_checks:
        raise ValueError(f"Data quality checks failed: {', '.join(failed_checks)}")
    
    logger.info("Data quality validation passed", results=results)
    return results


def monitor_signal_generation(**context):
    """Monitor signal generation performance."""
    import requests
    
    # Check signal generation service health
    signal_service_url = Variable.get("signal_service_url", default_var="http://signal-generator:8081")
    
    try:
        # Health check
        health_response = requests.get(f"{signal_service_url}/health", timeout=10)
        health_response.raise_for_status()
        
        # Get status
        status_response = requests.get(f"{signal_service_url}/status", timeout=10)
        status_response.raise_for_status()
        status_data = status_response.json()
        
        # Validate signal generation rate
        processing_stats = status_data.get('status', {}).get('processing_stats', {})
        signal_rate = processing_stats.get('signals_per_minute', 0)
        
        if signal_rate < 100:  # Less than 100 signals per minute
            logger.warning("Low signal generation rate", rate=signal_rate)
        
        logger.info("Signal generation monitoring completed", 
                   rate=signal_rate, 
                   status=status_data.get('status', {}).get('is_ready', False))
        
        return status_data
        
    except Exception as e:
        logger.error("Signal generation monitoring failed", error=str(e))
        raise


def cleanup_old_data(**context):
    """Clean up old data to manage storage."""
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    
    postgres = PostgresHook(postgres_conn_id='market_data_postgres')
    
    cleanup_queries = [
        # Delete price data older than 30 days
        """
        DELETE FROM market_data.price_data
        WHERE created_at < NOW() - INTERVAL '30 days'
        """,
        
        # Delete processed signals older than 90 days
        """
        DELETE FROM market_data.signals
        WHERE created_at < NOW() - INTERVAL '90 days'
        AND status = 'processed'
        """,
        
        # Delete old logs
        """
        DELETE FROM market_data.processing_logs
        WHERE created_at < NOW() - INTERVAL '7 days'
        """
    ]
    
    total_deleted = 0
    for query in cleanup_queries:
        try:
            result = postgres.run(query)
            logger.info("Cleanup query executed", query=query[:50] + "...")
            total_deleted += result if isinstance(result, int) else 0
        except Exception as e:
            logger.error("Cleanup query failed", query=query[:50] + "...", error=str(e))
    
    logger.info("Data cleanup completed", total_deleted=total_deleted)
    return total_deleted


# Task definitions

# Start task
start_pipeline = DummyOperator(
    task_id='start_pipeline',
    dag=dag
)

# Market hours check
market_hours_check = PythonOperator(
    task_id='check_market_hours',
    python_callable=check_market_hours,
    dag=dag
)

# Service health checks
with TaskGroup('health_checks', dag=dag) as health_checks_group:
    
    data_ingestion_health = HttpSensor(
        task_id='data_ingestion_health_check',
        http_conn_id='data_ingestion_service',
        endpoint='/health',
        timeout=30,
        poke_interval=10,
        mode='reschedule'
    )
    
    signal_generation_health = HttpSensor(
        task_id='signal_generation_health_check',
        http_conn_id='signal_generation_service',
        endpoint='/health',
        timeout=30,
        poke_interval=10,
        mode='reschedule'
    )
    
    risk_assessment_health = HttpSensor(
        task_id='risk_assessment_health_check',
        http_conn_id='risk_assessment_service',
        endpoint='/health',
        timeout=30,
        poke_interval=10,
        mode='reschedule'
    )
    
    notification_service_health = HttpSensor(
        task_id='notification_service_health_check',
        http_conn_id='notification_service',
        endpoint='/health',
        timeout=30,
        poke_interval=10,
        mode='reschedule'
    )

# Data quality validation
data_quality_validation = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag
)

# Signal generation monitoring
signal_monitoring = PythonOperator(
    task_id='monitor_signal_generation',
    python_callable=monitor_signal_generation,
    dag=dag
)

# Database maintenance
with TaskGroup('database_maintenance', dag=dag) as db_maintenance_group:
    
    # Update table statistics
    update_statistics = PostgresOperator(
        task_id='update_table_statistics',
        postgres_conn_id='market_data_postgres',
        sql="""
            ANALYZE market_data.price_data;
            ANALYZE market_data.signals;
            ANALYZE market_data.risk_scores;
            ANALYZE market_data.user_preferences;
        """,
    )
    
    # Vacuum tables
    vacuum_tables = PostgresOperator(
        task_id='vacuum_tables',
        postgres_conn_id='market_data_postgres',
        sql="""
            VACUUM ANALYZE market_data.price_data;
            VACUUM ANALYZE market_data.signals;
            VACUUM ANALYZE market_data.risk_scores;
        """,
    )
    
    # Cleanup old data
    cleanup_data = PythonOperator(
        task_id='cleanup_old_data',
        python_callable=cleanup_old_data,
    )

# System metrics collection
metrics_collection = BashOperator(
    task_id='collect_system_metrics',
    bash_command="""
    # Collect Kafka lag metrics
    kubectl exec -n market-scanning deployment/kafka -- kafka-consumer-groups.sh \
        --bootstrap-server localhost:9092 --describe --all-groups > /tmp/kafka_lag.txt
    
    # Collect service resource usage
    kubectl top pods -n market-scanning > /tmp/pod_resources.txt
    
    # Store metrics in monitoring system
    curl -X POST http://prometheus-pushgateway:9091/metrics/job/airflow-pipeline \
        -d "pipeline_execution_time $(date +%s)"
    """,
    dag=dag
)

# Alerting check
alerting_check = SimpleHttpOperator(
    task_id='check_alerting_system',
    http_conn_id='alertmanager',
    endpoint='/api/v1/alerts',
    method='GET',
    headers={'Content-Type': 'application/json'},
    dag=dag
)

# Performance optimization
performance_optimization = DockerOperator(
    task_id='run_performance_optimization',
    image='market-scanning/optimizer:latest',
    command='python -m optimizer.main --mode=daily',
    docker_url='unix://var/run/docker.sock',
    network_mode='market-scanning-network',
    environment={
        'POSTGRES_URL': '{{ var.value.postgres_url }}',
        'REDIS_URL': '{{ var.value.redis_url }}',
        'OPTIMIZATION_MODE': 'aggressive'
    },
    dag=dag
)

# End task
end_pipeline = DummyOperator(
    task_id='end_pipeline',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag
)

# Failure notification
failure_notification = SimpleHttpOperator(
    task_id='send_failure_notification',
    http_conn_id='notification_service',
    endpoint='/api/v1/alerts',
    method='POST',
    data={
        'alert_type': 'pipeline_failure',
        'severity': 'critical',
        'message': 'Market data pipeline failed',
        'timestamp': '{{ ts }}',
        'dag_run_id': '{{ dag_run.run_id }}'
    },
    headers={'Content-Type': 'application/json'},
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag
)

# Task dependencies
start_pipeline >> market_hours_check
market_hours_check >> health_checks_group
health_checks_group >> data_quality_validation
data_quality_validation >> signal_monitoring

# Parallel execution for maintenance tasks
signal_monitoring >> [db_maintenance_group, metrics_collection, alerting_check]
[db_maintenance_group, metrics_collection, alerting_check] >> performance_optimization

performance_optimization >> end_pipeline

# Failure path
[health_checks_group, data_quality_validation, signal_monitoring, 
 db_maintenance_group, metrics_collection, performance_optimization] >> failure_notification