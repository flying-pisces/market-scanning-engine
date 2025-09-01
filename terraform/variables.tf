# Terraform Variables for Market Scanning Engine Infrastructure

# Environment Configuration
variable "environment" {
  description = "Environment name (staging, production)"
  type        = string
  validation {
    condition = contains(["staging", "production"], var.environment)
    error_message = "Environment must be either 'staging' or 'production'."
  }
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
  validation {
    condition = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

# Kubernetes Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "enable_fargate" {
  description = "Enable Fargate profiles for serverless workloads"
  type        = bool
  default     = true
}

# Database Configuration
variable "db_master_username" {
  description = "Master username for RDS Aurora cluster"
  type        = string
  default     = "postgres"
}

variable "db_master_password" {
  description = "Master password for RDS Aurora cluster"
  type        = string
  sensitive   = true
  validation {
    condition = length(var.db_master_password) >= 8
    error_message = "Database password must be at least 8 characters long."
  }
}

variable "database_backup_retention_period" {
  description = "Backup retention period for database (days)"
  type        = number
  default     = 30
  validation {
    condition = var.database_backup_retention_period >= 7 && var.database_backup_retention_period <= 35
    error_message = "Backup retention period must be between 7 and 35 days."
  }
}

# Cache Configuration
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r6g.2xlarge"
}

variable "redis_num_cache_clusters" {
  description = "Number of cache clusters in Redis replication group"
  type        = number
  default     = 3
  validation {
    condition = var.redis_num_cache_clusters >= 2 && var.redis_num_cache_clusters <= 6
    error_message = "Number of cache clusters must be between 2 and 6."
  }
}

# Application Configuration
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

variable "ssl_certificate_arn" {
  description = "ARN of SSL certificate for load balancer"
  type        = string
  default     = ""
}

# Monitoring Configuration
variable "grafana_admin_password" {
  description = "Admin password for Grafana"
  type        = string
  sensitive   = true
  default     = "admin123!"
}

variable "enable_detailed_monitoring" {
  description = "Enable detailed CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch logs retention period (days)"
  type        = number
  default     = 30
  validation {
    condition = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention days must be a valid CloudWatch retention period."
  }
}

# Security Configuration
variable "enable_waf" {
  description = "Enable AWS WAF for application protection"
  type        = bool
  default     = true
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the application"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest for all services"
  type        = bool
  default     = true
}

# Scaling Configuration
variable "min_nodes" {
  description = "Minimum number of nodes in each node group"
  type        = number
  default     = 2
  validation {
    condition = var.min_nodes >= 1
    error_message = "Minimum nodes must be at least 1."
  }
}

variable "max_nodes" {
  description = "Maximum number of nodes in each node group"
  type        = number
  default     = 20
  validation {
    condition = var.max_nodes >= var.min_nodes
    error_message = "Maximum nodes must be greater than or equal to minimum nodes."
  }
}

variable "desired_nodes" {
  description = "Desired number of nodes in each node group"
  type        = number
  default     = 3
  validation {
    condition = var.desired_nodes >= var.min_nodes && var.desired_nodes <= var.max_nodes
    error_message = "Desired nodes must be between min_nodes and max_nodes."
  }
}

# High Availability Configuration
variable "multi_az" {
  description = "Enable multi-AZ deployment"
  type        = bool
  default     = true
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = false
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = true
}

variable "spot_instance_types" {
  description = "List of instance types for spot instances"
  type        = list(string)
  default     = ["c5.2xlarge", "c5.4xlarge", "r5.2xlarge", "r5.4xlarge"]
}

# Data Retention Configuration
variable "market_data_retention_days" {
  description = "Retention period for market data in S3 (days)"
  type        = number
  default     = 2555  # ~7 years
}

variable "log_data_retention_days" {
  description = "Retention period for application logs (days)"
  type        = number
  default     = 90
}

# Performance Configuration
variable "enable_enhanced_networking" {
  description = "Enable enhanced networking for EC2 instances"
  type        = bool
  default     = true
}

variable "enable_placement_groups" {
  description = "Enable placement groups for low latency"
  type        = bool
  default     = true
}

# Disaster Recovery Configuration
variable "backup_schedule" {
  description = "Cron expression for backup schedule"
  type        = string
  default     = "cron(0 2 * * ? *)"  # Daily at 2 AM UTC
}

variable "disaster_recovery_region" {
  description = "AWS region for disaster recovery"
  type        = string
  default     = "us-west-2"
}

# External Service Configuration
variable "external_api_keys" {
  description = "API keys for external data sources"
  type        = map(string)
  sensitive   = true
  default     = {}
}

# Feature Flags
variable "enable_kafka_msk" {
  description = "Use Amazon MSK instead of self-managed Kafka"
  type        = bool
  default     = true
}

variable "enable_opensearch" {
  description = "Enable Amazon OpenSearch for log analytics"
  type        = bool
  default     = false
}

variable "enable_timestream" {
  description = "Enable Amazon Timestream for time-series data"
  type        = bool
  default     = false
}

# Resource Tagging
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Local Variables for computed values
locals {
  # Environment-specific configurations
  environment_config = {
    staging = {
      instance_type           = "t3.medium"
      database_instance_class = "db.r6g.large"
      redis_node_type        = "cache.r6g.large"
      min_replicas           = 2
      max_replicas           = 10
    }
    production = {
      instance_type           = "c5.2xlarge"
      database_instance_class = "db.r6g.2xlarge"
      redis_node_type        = "cache.r6g.2xlarge"
      min_replicas           = 3
      max_replicas           = 50
    }
  }
  
  # Current environment configuration
  current_config = local.environment_config[var.environment]
  
  # Common tags
  common_tags = merge(
    {
      Environment   = var.environment
      Project       = "market-scanning-engine"
      ManagedBy     = "terraform"
      CostCenter    = "engineering"
      Backup        = "required"
      Monitoring    = "enabled"
      Security      = "high"
    },
    var.additional_tags
  )
}