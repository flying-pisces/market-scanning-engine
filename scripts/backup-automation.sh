#!/bin/bash
# Market Scanning Engine Automated Backup Script
# Production-grade backup automation with comprehensive error handling

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="market-scanning"
ENVIRONMENT="${ENVIRONMENT:-production}"
BACKUP_BUCKET="${BACKUP_BUCKET:-market-scanning-backups}"
DR_REGION="${DR_REGION:-us-west-2}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Logging setup
LOGFILE="/var/log/market-scanning/backup-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$(dirname "$LOGFILE")"

log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOGFILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_debug() { log "DEBUG" "$@"; }

# Error handling
trap 'handle_error $? $LINENO' ERR
trap 'cleanup' EXIT

handle_error() {
    local exit_code=$1
    local line_number=$2
    log_error "Script failed with exit code $exit_code at line $line_number"
    
    # Send alert
    send_alert "CRITICAL" "Backup failed" "Backup script failed at line $line_number with exit code $exit_code"
    
    exit $exit_code
}

cleanup() {
    log_info "Cleaning up temporary files"
    rm -rf /tmp/backup-$$*
}

# Utility functions
send_alert() {
    local severity=$1
    local title=$2
    local message=$3
    
    # Send to monitoring system
    curl -X POST "${ALERT_WEBHOOK_URL:-http://alertmanager:9093/api/v1/alerts}" \
        -H "Content-Type: application/json" \
        -d "[{
            \"labels\": {
                \"alertname\": \"BackupAlert\",
                \"severity\": \"$severity\",
                \"service\": \"backup-automation\",
                \"environment\": \"$ENVIRONMENT\"
            },
            \"annotations\": {
                \"title\": \"$title\",
                \"description\": \"$message\"
            }
        }]" || log_warn "Failed to send alert"
}

check_prerequisites() {
    log_info "Checking prerequisites"
    
    # Check required tools
    for tool in aws kubectl pg_dump redis-cli influx; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' not found"
            return 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        return 1
    fi
    
    # Check Kubernetes context
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Kubernetes cluster not accessible"
        return 1
    fi
    
    # Check S3 bucket access
    if ! aws s3 ls "s3://$BACKUP_BUCKET" &> /dev/null; then
        log_error "Cannot access backup bucket: $BACKUP_BUCKET"
        return 1
    fi
    
    log_info "Prerequisites check passed"
}

backup_database() {
    log_info "Starting database backup"
    
    local timestamp=$(date +%Y%m%d-%H%M%S)
    local backup_file="/tmp/backup-$$/db-$timestamp.sql"
    
    # Get database connection info from Kubernetes secrets
    local db_host=$(kubectl get secret postgres-credentials -n market-scanning -o jsonpath='{.data.host}' | base64 -d)
    local db_user=$(kubectl get secret postgres-credentials -n market-scanning -o jsonpath='{.data.username}' | base64 -d)
    local db_password=$(kubectl get secret postgres-credentials -n market-scanning -o jsonpath='{.data.password}' | base64 -d)
    local db_name=$(kubectl get secret postgres-credentials -n market-scanning -o jsonpath='{.data.database}' | base64 -d)
    
    # Create database backup
    log_info "Creating database dump"
    PGPASSWORD="$db_password" pg_dump \
        --host="$db_host" \
        --username="$db_user" \
        --dbname="$db_name" \
        --verbose \
        --clean \
        --create \
        --format=custom \
        --compress=9 \
        --file="$backup_file" || {
        log_error "Database backup failed"
        return 1
    }
    
    # Verify backup integrity
    log_info "Verifying backup integrity"
    if ! pg_restore --list "$backup_file" &> /dev/null; then
        log_error "Backup integrity check failed"
        return 1
    fi
    
    # Compress and encrypt backup
    local encrypted_file="${backup_file}.enc"
    log_info "Encrypting backup"
    openssl enc -aes-256-cbc -salt -pbkdf2 -iter 100000 \
        -in "$backup_file" \
        -out "$encrypted_file" \
        -pass "pass:$(aws ssm get-parameter --name "/market-scanning/$ENVIRONMENT/backup-key" --with-decryption --query 'Parameter.Value' --output text)"
    
    # Upload to S3
    local s3_path="s3://$BACKUP_BUCKET/database/$ENVIRONMENT/$(date +%Y/%m/%d)/db-$timestamp.sql.enc"
    log_info "Uploading backup to S3: $s3_path"
    aws s3 cp "$encrypted_file" "$s3_path" \
        --server-side-encryption aws:kms \
        --sse-kms-key-id "alias/market-scanning-backup" \
        --storage-class STANDARD_IA
    
    # Cross-region replication
    local dr_s3_path="s3://$BACKUP_BUCKET-dr/database/$ENVIRONMENT/$(date +%Y/%m/%d)/db-$timestamp.sql.enc"
    log_info "Replicating to DR region: $dr_s3_path"
    aws s3 cp "$s3_path" "$dr_s3_path" --region "$DR_REGION"
    
    # Create Aurora snapshot
    local snapshot_id="$PROJECT_NAME-$ENVIRONMENT-manual-$(date +%Y%m%d-%H%M%S)"
    log_info "Creating Aurora snapshot: $snapshot_id"
    aws rds create-db-cluster-snapshot \
        --db-cluster-identifier "$PROJECT_NAME-$ENVIRONMENT" \
        --db-cluster-snapshot-identifier "$snapshot_id"
    
    # Wait for snapshot completion
    log_info "Waiting for snapshot completion"
    aws rds wait db-cluster-snapshot-completed \
        --db-cluster-snapshot-identifier "$snapshot_id"
    
    # Copy snapshot to DR region
    log_info "Copying snapshot to DR region"
    aws rds copy-db-cluster-snapshot \
        --source-db-cluster-snapshot-identifier "$snapshot_id" \
        --target-db-cluster-snapshot-identifier "$snapshot_id-dr" \
        --source-region "$(aws configure get region)" \
        --region "$DR_REGION"
    
    log_info "Database backup completed successfully"
    
    # Update metrics
    send_metric "backup_database_duration" $(($(date +%s) - start_time)) "seconds"
    send_metric "backup_database_success" 1 "count"
    
    # Cleanup local files
    rm -f "$backup_file" "$encrypted_file"
}

backup_timeseries_data() {
    log_info "Starting time-series data backup"
    
    local timestamp=$(date +%Y%m%d-%H%M%S)
    local backup_dir="/tmp/backup-$$/influxdb-$timestamp"
    mkdir -p "$backup_dir"
    
    # Get InfluxDB connection info
    local influx_url=$(kubectl get secret influxdb-credentials -n market-scanning -o jsonpath='{.data.url}' | base64 -d)
    local influx_token=$(kubectl get secret influxdb-credentials -n market-scanning -o jsonpath='{.data.token}' | base64 -d)
    local influx_org=$(kubectl get secret influxdb-credentials -n market-scanning -o jsonpath='{.data.org}' | base64 -d)
    
    # Export data for each bucket
    for bucket in market_data signals risk_scores; do
        log_info "Backing up bucket: $bucket"
        
        # Export last 24 hours of data
        local start_time=$(date -d '24 hours ago' -u +%Y-%m-%dT%H:%M:%SZ)
        local end_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        
        influx query \
            --host "$influx_url" \
            --token "$influx_token" \
            --org "$influx_org" \
            "from(bucket: \"$bucket\")
             |> range(start: $start_time, stop: $end_time)
             |> pivot(rowKey:[\"_time\"], columnKey: [\"_field\"], valueColumn: \"_value\")" \
            --raw > "$backup_dir/$bucket.csv"
        
        # Compress the backup
        gzip "$backup_dir/$bucket.csv"
    done
    
    # Create archive
    local archive_file="/tmp/backup-$$/influxdb-$timestamp.tar.gz"
    tar -czf "$archive_file" -C "/tmp/backup-$$" "influxdb-$timestamp"
    
    # Encrypt archive
    local encrypted_file="${archive_file}.enc"
    openssl enc -aes-256-cbc -salt -pbkdf2 -iter 100000 \
        -in "$archive_file" \
        -out "$encrypted_file" \
        -pass "pass:$(aws ssm get-parameter --name "/market-scanning/$ENVIRONMENT/backup-key" --with-decryption --query 'Parameter.Value' --output text)"
    
    # Upload to S3
    local s3_path="s3://$BACKUP_BUCKET/timeseries/$ENVIRONMENT/$(date +%Y/%m/%d)/influxdb-$timestamp.tar.gz.enc"
    log_info "Uploading time-series backup to S3: $s3_path"
    aws s3 cp "$encrypted_file" "$s3_path" \
        --server-side-encryption aws:kms \
        --sse-kms-key-id "alias/market-scanning-backup"
    
    # Cross-region replication
    aws s3 cp "$s3_path" "s3://$BACKUP_BUCKET-dr/timeseries/$ENVIRONMENT/$(date +%Y/%m/%d)/influxdb-$timestamp.tar.gz.enc" \
        --region "$DR_REGION"
    
    log_info "Time-series backup completed successfully"
    
    # Cleanup
    rm -rf "$backup_dir" "$archive_file" "$encrypted_file"
}

backup_redis_cache() {
    log_info "Starting Redis cache backup"
    
    local timestamp=$(date +%Y%m%d-%H%M%S)
    
    # Get Redis connection info
    local redis_host=$(kubectl get secret redis-credentials -n market-scanning -o jsonpath='{.data.host}' | base64 -d)
    local redis_port=$(kubectl get secret redis-credentials -n market-scanning -o jsonpath='{.data.port}' | base64 -d)
    local redis_password=$(kubectl get secret redis-credentials -n market-scanning -o jsonpath='{.data.password}' | base64 -d)
    
    # Create Redis backup
    local backup_file="/tmp/backup-$$/redis-$timestamp.rdb"
    log_info "Creating Redis backup"
    
    # Use BGSAVE for non-blocking backup
    redis-cli -h "$redis_host" -p "$redis_port" -a "$redis_password" BGSAVE
    
    # Wait for backup completion
    while [ "$(redis-cli -h "$redis_host" -p "$redis_port" -a "$redis_password" LASTSAVE)" -eq "$(redis-cli -h "$redis_host" -p "$redis_port" -a "$redis_password" LASTSAVE)" ]; do
        sleep 1
    done
    
    # Copy RDB file
    kubectl cp "redis-0:/data/dump.rdb" "$backup_file" -n market-scanning
    
    # Compress and encrypt
    local compressed_file="${backup_file}.gz"
    gzip "$backup_file"
    
    local encrypted_file="${compressed_file}.enc"
    openssl enc -aes-256-cbc -salt -pbkdf2 -iter 100000 \
        -in "$compressed_file" \
        -out "$encrypted_file" \
        -pass "pass:$(aws ssm get-parameter --name "/market-scanning/$ENVIRONMENT/backup-key" --with-decryption --query 'Parameter.Value' --output text)"
    
    # Upload to S3
    local s3_path="s3://$BACKUP_BUCKET/redis/$ENVIRONMENT/$(date +%Y/%m/%d)/redis-$timestamp.rdb.gz.enc"
    log_info "Uploading Redis backup to S3: $s3_path"
    aws s3 cp "$encrypted_file" "$s3_path" \
        --server-side-encryption aws:kms \
        --sse-kms-key-id "alias/market-scanning-backup"
    
    # Cross-region replication
    aws s3 cp "$s3_path" "s3://$BACKUP_BUCKET-dr/redis/$ENVIRONMENT/$(date +%Y/%m/%d)/redis-$timestamp.rdb.gz.enc" \
        --region "$DR_REGION"
    
    log_info "Redis backup completed successfully"
    
    # Cleanup
    rm -f "$compressed_file" "$encrypted_file"
}

backup_kafka_config() {
    log_info "Starting Kafka configuration backup"
    
    local timestamp=$(date +%Y%m%d-%H%M%S)
    local backup_dir="/tmp/backup-$$/kafka-$timestamp"
    mkdir -p "$backup_dir"
    
    # Get Kafka bootstrap servers
    local kafka_brokers=$(kubectl get service kafka-service -n market-scanning -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'):9092
    
    # Backup topic configurations
    log_info "Backing up topic configurations"
    kafka-topics --bootstrap-server "$kafka_brokers" --list > "$backup_dir/topics.txt"
    
    while IFS= read -r topic; do
        kafka-configs --bootstrap-server "$kafka_brokers" \
            --entity-type topics --entity-name "$topic" \
            --describe > "$backup_dir/topic-config-$topic.txt"
    done < "$backup_dir/topics.txt"
    
    # Backup consumer group information
    log_info "Backing up consumer groups"
    kafka-consumer-groups --bootstrap-server "$kafka_brokers" --list > "$backup_dir/consumer-groups.txt"
    
    while IFS= read -r group; do
        kafka-consumer-groups --bootstrap-server "$kafka_brokers" \
            --group "$group" --describe > "$backup_dir/consumer-group-$group.txt" || true
    done < "$backup_dir/consumer-groups.txt"
    
    # Create archive
    local archive_file="/tmp/backup-$$/kafka-config-$timestamp.tar.gz"
    tar -czf "$archive_file" -C "/tmp/backup-$$" "kafka-$timestamp"
    
    # Upload to S3
    local s3_path="s3://$BACKUP_BUCKET/kafka-config/$ENVIRONMENT/$(date +%Y/%m/%d)/kafka-config-$timestamp.tar.gz"
    log_info "Uploading Kafka config backup to S3: $s3_path"
    aws s3 cp "$archive_file" "$s3_path"
    
    # Cross-region replication
    aws s3 cp "$s3_path" "s3://$BACKUP_BUCKET-dr/kafka-config/$ENVIRONMENT/$(date +%Y/%m/%d)/kafka-config-$timestamp.tar.gz" \
        --region "$DR_REGION"
    
    log_info "Kafka configuration backup completed successfully"
    
    # Cleanup
    rm -rf "$backup_dir" "$archive_file"
}

backup_kubernetes_configs() {
    log_info "Starting Kubernetes configuration backup"
    
    local timestamp=$(date +%Y%m%d-%H%M%S)
    local backup_dir="/tmp/backup-$$/k8s-$timestamp"
    mkdir -p "$backup_dir"
    
    # Backup all resources in market-scanning namespace
    log_info "Backing up Kubernetes resources"
    
    # Get all resource types
    local resources=$(kubectl api-resources --verbs=list --namespaced -o name | grep -v events | tr '\n' ',')
    
    # Export resources
    kubectl get "$resources" -n market-scanning -o yaml > "$backup_dir/all-resources.yaml"
    
    # Export secrets separately (they need special handling)
    kubectl get secrets -n market-scanning -o yaml > "$backup_dir/secrets.yaml"
    
    # Export configmaps
    kubectl get configmaps -n market-scanning -o yaml > "$backup_dir/configmaps.yaml"
    
    # Export persistent volumes and claims
    kubectl get pv,pvc -n market-scanning -o yaml > "$backup_dir/storage.yaml"
    
    # Create archive
    local archive_file="/tmp/backup-$$/k8s-config-$timestamp.tar.gz"
    tar -czf "$archive_file" -C "/tmp/backup-$$" "k8s-$timestamp"
    
    # Encrypt archive
    local encrypted_file="${archive_file}.enc"
    openssl enc -aes-256-cbc -salt -pbkdf2 -iter 100000 \
        -in "$archive_file" \
        -out "$encrypted_file" \
        -pass "pass:$(aws ssm get-parameter --name "/market-scanning/$ENVIRONMENT/backup-key" --with-decryption --query 'Parameter.Value' --output text)"
    
    # Upload to S3
    local s3_path="s3://$BACKUP_BUCKET/kubernetes/$ENVIRONMENT/$(date +%Y/%m/%d)/k8s-config-$timestamp.tar.gz.enc"
    log_info "Uploading Kubernetes config backup to S3: $s3_path"
    aws s3 cp "$encrypted_file" "$s3_path" \
        --server-side-encryption aws:kms \
        --sse-kms-key-id "alias/market-scanning-backup"
    
    # Cross-region replication
    aws s3 cp "$s3_path" "s3://$BACKUP_BUCKET-dr/kubernetes/$ENVIRONMENT/$(date +%Y/%m/%d)/k8s-config-$timestamp.tar.gz.enc" \
        --region "$DR_REGION"
    
    log_info "Kubernetes configuration backup completed successfully"
    
    # Cleanup
    rm -rf "$backup_dir" "$archive_file" "$encrypted_file"
}

cleanup_old_backups() {
    log_info "Cleaning up old backups"
    
    # Calculate cutoff date
    local cutoff_date=$(date -d "$RETENTION_DAYS days ago" +%Y-%m-%d)
    
    # Cleanup S3 backups
    for prefix in database timeseries redis kafka-config kubernetes; do
        log_info "Cleaning up old $prefix backups"
        
        aws s3api list-objects-v2 \
            --bucket "$BACKUP_BUCKET" \
            --prefix "$prefix/$ENVIRONMENT/" \
            --query "Contents[?LastModified<'$cutoff_date'].Key" \
            --output text | while read -r key; do
            if [ -n "$key" ]; then
                log_info "Deleting old backup: s3://$BACKUP_BUCKET/$key"
                aws s3 rm "s3://$BACKUP_BUCKET/$key"
            fi
        done
    done
    
    # Cleanup Aurora snapshots
    log_info "Cleaning up old Aurora snapshots"
    aws rds describe-db-cluster-snapshots \
        --db-cluster-identifier "$PROJECT_NAME-$ENVIRONMENT" \
        --snapshot-type manual \
        --query "DBClusterSnapshots[?SnapshotCreateTime<'$cutoff_date'].DBClusterSnapshotIdentifier" \
        --output text | while read -r snapshot_id; do
        if [ -n "$snapshot_id" ]; then
            log_info "Deleting old snapshot: $snapshot_id"
            aws rds delete-db-cluster-snapshot \
                --db-cluster-snapshot-identifier "$snapshot_id"
        fi
    done
}

send_metric() {
    local metric_name=$1
    local value=$2
    local unit=$3
    
    # Send metric to CloudWatch
    aws cloudwatch put-metric-data \
        --namespace "MarketScanning/Backup" \
        --metric-data "MetricName=$metric_name,Value=$value,Unit=$unit,Dimensions=[{Name=Environment,Value=$ENVIRONMENT}]"
}

generate_backup_report() {
    log_info "Generating backup report"
    
    local report_file="/tmp/backup-report-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "backup_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$ENVIRONMENT",
    "components": {
        "database": {
            "status": "completed",
            "size": "$(aws s3api head-object --bucket $BACKUP_BUCKET --key database/$ENVIRONMENT/$(date +%Y/%m/%d)/latest.sql.enc --query ContentLength --output text 2>/dev/null || echo 0)"
        },
        "timeseries": {
            "status": "completed"
        },
        "redis": {
            "status": "completed"
        },
        "kafka_config": {
            "status": "completed"
        },
        "kubernetes": {
            "status": "completed"
        }
    },
    "metrics": {
        "total_duration": $(($(date +%s) - start_time)),
        "success_rate": 100
    }
}
EOF
    
    # Upload report
    aws s3 cp "$report_file" "s3://$BACKUP_BUCKET/reports/$ENVIRONMENT/$(date +%Y/%m/%d)/backup-report-$(date +%Y%m%d-%H%M%S).json"
    
    # Send notification
    send_alert "INFO" "Backup Completed" "Daily backup completed successfully for $ENVIRONMENT environment"
    
    rm -f "$report_file"
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    log_info "Starting backup process for $ENVIRONMENT environment"
    
    # Check prerequisites
    check_prerequisites
    
    # Perform backups
    backup_database
    backup_timeseries_data
    backup_redis_cache
    backup_kafka_config
    backup_kubernetes_configs
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Generate report
    generate_backup_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_info "Backup process completed successfully in $duration seconds"
    
    # Send success metrics
    send_metric "backup_total_duration" $duration "seconds"
    send_metric "backup_success" 1 "count"
}

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi