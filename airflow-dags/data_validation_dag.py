"""
Data Validation DAG
Comprehensive data quality monitoring and validation for the market scanning engine.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup

import structlog
import pandas as pd
import numpy as np

logger = structlog.get_logger(__name__)

# DAG configuration
default_args = {
    'owner': 'data-quality-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

dag = DAG(
    'data_validation_pipeline',
    default_args=default_args,
    description='Comprehensive data quality validation and monitoring',
    schedule_interval=timedelta(minutes=5),  # Run every 5 minutes
    catchup=False,
    max_active_runs=1,
    tags=['data-quality', 'validation', 'monitoring']
)


def run_data_freshness_checks(**context):
    """Check data freshness across all sources."""
    postgres = PostgresHook(postgres_conn_id='market_data_postgres')
    
    freshness_queries = {
        'equity_data': """
            SELECT 
                'equity_data' as source,
                COUNT(*) as record_count,
                MAX(timestamp) as latest_timestamp,
                EXTRACT(EPOCH FROM (NOW() - MAX(timestamp))) as seconds_since_latest
            FROM market_data.equity_prices
            WHERE timestamp >= NOW() - INTERVAL '10 minutes'
        """,
        
        'options_data': """
            SELECT 
                'options_data' as source,
                COUNT(*) as record_count,
                MAX(timestamp) as latest_timestamp,
                EXTRACT(EPOCH FROM (NOW() - MAX(timestamp))) as seconds_since_latest
            FROM market_data.options_data
            WHERE timestamp >= NOW() - INTERVAL '10 minutes'
        """,
        
        'news_data': """
            SELECT 
                'news_data' as source,
                COUNT(*) as record_count,
                MAX(published_at) as latest_timestamp,
                EXTRACT(EPOCH FROM (NOW() - MAX(published_at))) as seconds_since_latest
            FROM market_data.news_articles
            WHERE published_at >= NOW() - INTERVAL '30 minutes'
        """,
        
        'economic_indicators': """
            SELECT 
                'economic_indicators' as source,
                COUNT(*) as record_count,
                MAX(timestamp) as latest_timestamp,
                EXTRACT(EPOCH FROM (NOW() - MAX(timestamp))) as seconds_since_latest
            FROM market_data.economic_indicators
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
        """
    }
    
    freshness_results = {}
    alerts = []
    
    # Freshness thresholds (in seconds)
    thresholds = {
        'equity_data': 300,    # 5 minutes
        'options_data': 600,   # 10 minutes
        'news_data': 1800,     # 30 minutes
        'economic_indicators': 3600  # 1 hour
    }
    
    for source, query in freshness_queries.items():
        try:
            result = postgres.get_first(query)
            if result:
                source_name, count, latest_ts, seconds_since = result
                freshness_results[source] = {
                    'record_count': count,
                    'latest_timestamp': str(latest_ts),
                    'seconds_since_latest': seconds_since,
                    'threshold': thresholds[source],
                    'is_fresh': seconds_since < thresholds[source]
                }
                
                if seconds_since > thresholds[source]:
                    alerts.append(f"{source}: Data is {seconds_since:.0f}s old (threshold: {thresholds[source]}s)")
                    
            else:
                freshness_results[source] = {'error': 'No data found'}
                alerts.append(f"{source}: No recent data found")
                
        except Exception as e:
            logger.error(f"Freshness check failed for {source}", error=str(e))
            freshness_results[source] = {'error': str(e)}
            alerts.append(f"{source}: Check failed - {str(e)}")
    
    # Store results
    context['task_instance'].xcom_push(key='freshness_results', value=freshness_results)
    context['task_instance'].xcom_push(key='freshness_alerts', value=alerts)
    
    logger.info("Data freshness check completed", 
               results=freshness_results, 
               alert_count=len(alerts))
    
    return freshness_results


def run_data_completeness_checks(**context):
    """Check data completeness and missing values."""
    postgres = PostgresHook(postgres_conn_id='market_data_postgres')
    
    completeness_queries = {
        'equity_missing_prices': """
            SELECT COUNT(*) as missing_count
            FROM market_data.equity_prices
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
            AND (price IS NULL OR price <= 0 OR volume IS NULL)
        """,
        
        'options_missing_greeks': """
            SELECT COUNT(*) as missing_count
            FROM market_data.options_data
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
            AND (delta IS NULL OR gamma IS NULL OR theta IS NULL OR vega IS NULL)
        """,
        
        'symbols_without_data': """
            SELECT COUNT(DISTINCT symbol) as missing_symbols
            FROM market_data.symbols s
            LEFT JOIN market_data.equity_prices p ON s.symbol = p.symbol
                AND p.timestamp >= NOW() - INTERVAL '30 minutes'
            WHERE p.symbol IS NULL
            AND s.is_active = true
        """,
        
        'incomplete_news_articles': """
            SELECT COUNT(*) as incomplete_count
            FROM market_data.news_articles
            WHERE published_at >= NOW() - INTERVAL '2 hours'
            AND (title IS NULL OR content IS NULL OR source IS NULL)
        """
    }
    
    completeness_results = {}
    alerts = []
    
    # Completeness thresholds
    thresholds = {
        'equity_missing_prices': 10,
        'options_missing_greeks': 50,
        'symbols_without_data': 5,
        'incomplete_news_articles': 2
    }
    
    for check, query in completeness_queries.items():
        try:
            result = postgres.get_first(query)
            missing_count = result[0] if result else 0
            
            completeness_results[check] = {
                'missing_count': missing_count,
                'threshold': thresholds[check],
                'is_acceptable': missing_count <= thresholds[check]
            }
            
            if missing_count > thresholds[check]:
                alerts.append(f"{check}: {missing_count} missing (threshold: {thresholds[check]})")
                
        except Exception as e:
            logger.error(f"Completeness check failed for {check}", error=str(e))
            completeness_results[check] = {'error': str(e)}
            alerts.append(f"{check}: Check failed - {str(e)}")
    
    context['task_instance'].xcom_push(key='completeness_results', value=completeness_results)
    context['task_instance'].xcom_push(key='completeness_alerts', value=alerts)
    
    logger.info("Data completeness check completed", 
               results=completeness_results, 
               alert_count=len(alerts))
    
    return completeness_results


def run_data_accuracy_checks(**context):
    """Check data accuracy and detect anomalies."""
    postgres = PostgresHook(postgres_conn_id='market_data_postgres')
    
    accuracy_queries = {
        'price_anomalies': """
            SELECT COUNT(*) as anomaly_count
            FROM market_data.equity_prices
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
            AND (
                price > (SELECT AVG(price) * 3 FROM market_data.equity_prices p2 
                        WHERE p2.symbol = equity_prices.symbol 
                        AND p2.timestamp >= NOW() - INTERVAL '24 hours')
                OR price < (SELECT AVG(price) * 0.1 FROM market_data.equity_prices p3 
                           WHERE p3.symbol = equity_prices.symbol 
                           AND p3.timestamp >= NOW() - INTERVAL '24 hours')
            )
        """,
        
        'volume_spikes': """
            SELECT COUNT(*) as spike_count
            FROM market_data.equity_prices
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
            AND volume > (
                SELECT AVG(volume) * 10 
                FROM market_data.equity_prices p2 
                WHERE p2.symbol = equity_prices.symbol 
                AND p2.timestamp >= NOW() - INTERVAL '7 days'
            )
        """,
        
        'duplicate_records': """
            SELECT COUNT(*) as duplicate_count
            FROM (
                SELECT symbol, timestamp, COUNT(*)
                FROM market_data.equity_prices
                WHERE timestamp >= NOW() - INTERVAL '2 hours'
                GROUP BY symbol, timestamp
                HAVING COUNT(*) > 1
            ) t
        """,
        
        'invalid_options_data': """
            SELECT COUNT(*) as invalid_count
            FROM market_data.options_data
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
            AND (
                strike_price <= 0 
                OR expiration_date <= CURRENT_DATE
                OR implied_volatility < 0 
                OR implied_volatility > 5
            )
        """
    }
    
    accuracy_results = {}
    alerts = []
    
    # Accuracy thresholds
    thresholds = {
        'price_anomalies': 5,
        'volume_spikes': 20,
        'duplicate_records': 0,
        'invalid_options_data': 10
    }
    
    for check, query in accuracy_queries.items():
        try:
            result = postgres.get_first(query)
            count = result[0] if result else 0
            
            accuracy_results[check] = {
                'count': count,
                'threshold': thresholds[check],
                'is_acceptable': count <= thresholds[check]
            }
            
            if count > thresholds[check]:
                alerts.append(f"{check}: {count} issues found (threshold: {thresholds[check]})")
                
        except Exception as e:
            logger.error(f"Accuracy check failed for {check}", error=str(e))
            accuracy_results[check] = {'error': str(e)}
            alerts.append(f"{check}: Check failed - {str(e)}")
    
    context['task_instance'].xcom_push(key='accuracy_results', value=accuracy_results)
    context['task_instance'].xcom_push(key='accuracy_alerts', value=alerts)
    
    logger.info("Data accuracy check completed", 
               results=accuracy_results, 
               alert_count=len(alerts))
    
    return accuracy_results


def run_schema_validation(**context):
    """Validate data schemas and constraints."""
    postgres = PostgresHook(postgres_conn_id='market_data_postgres')
    
    schema_queries = {
        'constraint_violations': """
            SELECT 
                schemaname, tablename, conname as constraint_name
            FROM pg_constraint pc
            JOIN pg_class c ON c.oid = pc.conrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE NOT pg_constraint_is_valid(pc.oid)
            AND schemaname = 'market_data'
        """,
        
        'null_violations': """
            SELECT COUNT(*) as violations
            FROM information_schema.columns c
            LEFT JOIN (
                -- Check for null violations in not-null columns
                SELECT 'equity_prices' as table_name, 'symbol' as column_name, 
                       SUM(CASE WHEN symbol IS NULL THEN 1 ELSE 0 END) as null_count
                FROM market_data.equity_prices
                WHERE timestamp >= NOW() - INTERVAL '1 hour'
            ) v ON v.table_name = c.table_name AND v.column_name = c.column_name
            WHERE c.table_schema = 'market_data'
            AND c.is_nullable = 'NO'
            AND COALESCE(v.null_count, 0) > 0
        """,
        
        'data_type_violations': """
            SELECT COUNT(*) as violations
            FROM market_data.equity_prices
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
            AND (
                NOT (price ~ '^[0-9]*\.?[0-9]+$')
                OR NOT (volume ~ '^[0-9]+$')
            )
        """
    }
    
    schema_results = {}
    alerts = []
    
    for check, query in schema_queries.items():
        try:
            if check == 'constraint_violations':
                results = postgres.get_records(query)
                violation_count = len(results) if results else 0
                schema_results[check] = {
                    'violation_count': violation_count,
                    'violations': results[:10] if results else []  # First 10 violations
                }
                
                if violation_count > 0:
                    alerts.append(f"{check}: {violation_count} constraint violations found")
            else:
                result = postgres.get_first(query)
                violation_count = result[0] if result else 0
                schema_results[check] = {
                    'violation_count': violation_count,
                    'is_acceptable': violation_count == 0
                }
                
                if violation_count > 0:
                    alerts.append(f"{check}: {violation_count} violations found")
                    
        except Exception as e:
            logger.error(f"Schema validation failed for {check}", error=str(e))
            schema_results[check] = {'error': str(e)}
            alerts.append(f"{check}: Check failed - {str(e)}")
    
    context['task_instance'].xcom_push(key='schema_results', value=schema_results)
    context['task_instance'].xcom_push(key='schema_alerts', value=alerts)
    
    logger.info("Schema validation completed", 
               results=schema_results, 
               alert_count=len(alerts))
    
    return schema_results


def assess_data_quality_score(**context):
    """Calculate overall data quality score."""
    ti = context['task_instance']
    
    # Get results from previous tasks
    freshness_results = ti.xcom_pull(key='freshness_results', task_ids='data_freshness_checks') or {}
    completeness_results = ti.xcom_pull(key='completeness_results', task_ids='data_completeness_checks') or {}
    accuracy_results = ti.xcom_pull(key='accuracy_results', task_ids='data_accuracy_checks') or {}
    schema_results = ti.xcom_pull(key='schema_results', task_ids='schema_validation') or {}
    
    # Calculate scores (0-100)
    freshness_score = 0
    if freshness_results:
        fresh_sources = sum(1 for r in freshness_results.values() 
                          if isinstance(r, dict) and r.get('is_fresh', False))
        total_sources = len([r for r in freshness_results.values() 
                           if isinstance(r, dict) and 'is_fresh' in r])
        freshness_score = (fresh_sources / total_sources * 100) if total_sources > 0 else 0
    
    completeness_score = 0
    if completeness_results:
        acceptable_checks = sum(1 for r in completeness_results.values() 
                              if isinstance(r, dict) and r.get('is_acceptable', False))
        total_checks = len([r for r in completeness_results.values() 
                          if isinstance(r, dict) and 'is_acceptable' in r])
        completeness_score = (acceptable_checks / total_checks * 100) if total_checks > 0 else 0
    
    accuracy_score = 0
    if accuracy_results:
        acceptable_checks = sum(1 for r in accuracy_results.values() 
                              if isinstance(r, dict) and r.get('is_acceptable', False))
        total_checks = len([r for r in accuracy_results.values() 
                          if isinstance(r, dict) and 'is_acceptable' in r])
        accuracy_score = (acceptable_checks / total_checks * 100) if total_checks > 0 else 0
    
    schema_score = 0
    if schema_results:
        clean_checks = sum(1 for r in schema_results.values() 
                         if isinstance(r, dict) and r.get('violation_count', 1) == 0)
        total_checks = len([r for r in schema_results.values() 
                          if isinstance(r, dict) and 'violation_count' in r])
        schema_score = (clean_checks / total_checks * 100) if total_checks > 0 else 0
    
    # Weighted overall score
    overall_score = (
        freshness_score * 0.3 +
        completeness_score * 0.25 +
        accuracy_score * 0.3 +
        schema_score * 0.15
    )
    
    quality_assessment = {
        'overall_score': round(overall_score, 2),
        'freshness_score': round(freshness_score, 2),
        'completeness_score': round(completeness_score, 2),
        'accuracy_score': round(accuracy_score, 2),
        'schema_score': round(schema_score, 2),
        'assessment_timestamp': datetime.now().isoformat(),
        'quality_level': 'excellent' if overall_score >= 90 else
                        'good' if overall_score >= 75 else
                        'acceptable' if overall_score >= 60 else
                        'poor'
    }
    
    # Store assessment in database
    postgres = PostgresHook(postgres_conn_id='market_data_postgres')
    postgres.run("""
        INSERT INTO market_data.data_quality_scores 
        (timestamp, overall_score, freshness_score, completeness_score, accuracy_score, schema_score, quality_level)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, parameters=(
        datetime.now(),
        quality_assessment['overall_score'],
        quality_assessment['freshness_score'],
        quality_assessment['completeness_score'],
        quality_assessment['accuracy_score'],
        quality_assessment['schema_score'],
        quality_assessment['quality_level']
    ))
    
    logger.info("Data quality assessment completed", assessment=quality_assessment)
    
    return quality_assessment


def determine_alert_level(**context):
    """Determine if alerts need to be sent based on quality score."""
    ti = context['task_instance']
    quality_assessment = ti.xcom_pull(task_ids='assess_data_quality')
    
    if not quality_assessment:
        return 'send_critical_alert'
    
    overall_score = quality_assessment.get('overall_score', 0)
    
    if overall_score < 50:
        return 'send_critical_alert'
    elif overall_score < 75:
        return 'send_warning_alert'
    else:
        return 'no_alert_needed'


# Task definitions
start_validation = DummyOperator(
    task_id='start_validation',
    dag=dag
)

# Data quality checks
data_freshness_checks = PythonOperator(
    task_id='data_freshness_checks',
    python_callable=run_data_freshness_checks,
    dag=dag
)

data_completeness_checks = PythonOperator(
    task_id='data_completeness_checks',
    python_callable=run_data_completeness_checks,
    dag=dag
)

data_accuracy_checks = PythonOperator(
    task_id='data_accuracy_checks',
    python_callable=run_data_accuracy_checks,
    dag=dag
)

schema_validation = PythonOperator(
    task_id='schema_validation',
    python_callable=run_schema_validation,
    dag=dag
)

# Quality assessment
quality_assessment = PythonOperator(
    task_id='assess_data_quality',
    python_callable=assess_data_quality_score,
    dag=dag
)

# Alert decision
alert_decision = BranchPythonOperator(
    task_id='determine_alert_level',
    python_callable=determine_alert_level,
    dag=dag
)

# Alert tasks
critical_alert = SimpleHttpOperator(
    task_id='send_critical_alert',
    http_conn_id='notification_service',
    endpoint='/api/v1/alerts',
    method='POST',
    data={
        'alert_type': 'data_quality_critical',
        'severity': 'critical',
        'message': 'Critical data quality issues detected',
        'timestamp': '{{ ts }}'
    },
    headers={'Content-Type': 'application/json'},
    dag=dag
)

warning_alert = SimpleHttpOperator(
    task_id='send_warning_alert',
    http_conn_id='notification_service',
    endpoint='/api/v1/alerts',
    method='POST',
    data={
        'alert_type': 'data_quality_warning',
        'severity': 'warning',
        'message': 'Data quality issues detected',
        'timestamp': '{{ ts }}'
    },
    headers={'Content-Type': 'application/json'},
    dag=dag
)

no_alert = DummyOperator(
    task_id='no_alert_needed',
    dag=dag
)

end_validation = DummyOperator(
    task_id='end_validation',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag
)

# Task dependencies
start_validation >> [data_freshness_checks, data_completeness_checks, data_accuracy_checks, schema_validation]

[data_freshness_checks, data_completeness_checks, data_accuracy_checks, schema_validation] >> quality_assessment

quality_assessment >> alert_decision
alert_decision >> [critical_alert, warning_alert, no_alert]

[critical_alert, warning_alert, no_alert] >> end_validation