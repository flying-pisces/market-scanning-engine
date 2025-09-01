# Main Terraform configuration for Market Scanning Engine Infrastructure
# Production-ready, highly available, multi-region deployment

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  # Remote state management
  backend "s3" {
    bucket         = "market-scanning-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-locks"
  }
}

# Local variables
locals {
  project_name = "market-scanning"
  environment  = var.environment
  region       = var.aws_region
  
  common_tags = {
    Project     = local.project_name
    Environment = local.environment
    ManagedBy   = "terraform"
    CostCenter  = "engineering"
    Owner       = "platform-team"
  }
  
  # Cluster configuration
  cluster_name = "${local.project_name}-${local.environment}"
  node_groups = {
    system = {
      instance_types = ["t3.medium"]
      scaling_config = {
        desired_size = 2
        max_size     = 4
        min_size     = 2
      }
      capacity_type = "ON_DEMAND"
      taints = [{
        key    = "system"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
    
    data_processing = {
      instance_types = ["c5.2xlarge", "c5.4xlarge"]
      scaling_config = {
        desired_size = 3
        max_size     = 20
        min_size     = 3
      }
      capacity_type = "SPOT"
      labels = {
        workload = "data-processing"
      }
    }
    
    real_time = {
      instance_types = ["r5.xlarge", "r5.2xlarge"]
      scaling_config = {
        desired_size = 2
        max_size     = 10
        min_size     = 2
      }
      capacity_type = "ON_DEMAND"
      labels = {
        workload = "real-time"
      }
      taints = [{
        key    = "real-time"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Module
module "vpc" {
  source = "./modules/vpc"
  
  project_name        = local.project_name
  environment        = local.environment
  region             = local.region
  availability_zones = data.aws_availability_zones.available.names
  
  vpc_cidr = var.vpc_cidr
  
  # Network segmentation for high-frequency trading
  private_subnets = [
    cidrsubnet(var.vpc_cidr, 4, 1),  # 10.0.16.0/20
    cidrsubnet(var.vpc_cidr, 4, 2),  # 10.0.32.0/20
    cidrsubnet(var.vpc_cidr, 4, 3),  # 10.0.48.0/20
  ]
  
  public_subnets = [
    cidrsubnet(var.vpc_cidr, 8, 1),  # 10.0.1.0/24
    cidrsubnet(var.vpc_cidr, 8, 2),  # 10.0.2.0/24
    cidrsubnet(var.vpc_cidr, 8, 3),  # 10.0.3.0/24
  ]
  
  database_subnets = [
    cidrsubnet(var.vpc_cidr, 8, 21), # 10.0.21.0/24
    cidrsubnet(var.vpc_cidr, 8, 22), # 10.0.22.0/24
    cidrsubnet(var.vpc_cidr, 8, 23), # 10.0.23.0/24
  ]
  
  tags = local.common_tags
}

# EKS Cluster Module
module "eks" {
  source = "./modules/eks"
  
  project_name = local.project_name
  environment  = local.environment
  cluster_name = local.cluster_name
  
  vpc_id               = module.vpc.vpc_id
  private_subnet_ids   = module.vpc.private_subnet_ids
  public_subnet_ids    = module.vpc.public_subnet_ids
  
  cluster_version = var.kubernetes_version
  
  # Enhanced cluster configuration for financial workloads
  cluster_encryption_config = [{
    provider_key_arn = module.kms.cluster_key_arn
    resources        = ["secrets"]
  }]
  
  cluster_enabled_log_types = [
    "api", "audit", "authenticator", "controllerManager", "scheduler"
  ]
  
  node_groups = local.node_groups
  
  # Fargate profiles for serverless workloads
  fargate_profiles = {
    monitoring = {
      selectors = [{
        namespace = "monitoring"
        labels = {
          fargate = "true"
        }
      }]
    }
    
    airflow = {
      selectors = [{
        namespace = "airflow"
      }]
    }
  }
  
  tags = local.common_tags
}

# Database Module (RDS Aurora with Read Replicas)
module "database" {
  source = "./modules/database"
  
  project_name = local.project_name
  environment  = local.environment
  
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.database_subnet_ids
  vpc_security_group_ids = [module.vpc.database_security_group_id]
  
  # Aurora PostgreSQL cluster for ACID compliance
  engine         = "aurora-postgresql"
  engine_version = "15.4"
  
  master_username = var.db_master_username
  master_password = var.db_master_password
  
  # High availability configuration
  instances = {
    writer = {
      instance_class      = "db.r6g.2xlarge"
      publicly_accessible = false
    }
    reader_1 = {
      instance_class      = "db.r6g.xlarge"
      publicly_accessible = false
    }
    reader_2 = {
      instance_class      = "db.r6g.xlarge"
      publicly_accessible = false
    }
  }
  
  # Performance optimization
  cluster_parameters = [
    {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements,auto_explain"
    },
    {
      name  = "log_statement"
      value = "all"
    },
    {
      name  = "log_min_duration_statement"
      value = "1000"  # Log slow queries > 1s
    }
  ]
  
  # Backup and maintenance
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Encryption
  kms_key_id = module.kms.database_key_arn
  
  tags = local.common_tags
}

# Cache Module (ElastiCache Redis Cluster)
module "cache" {
  source = "./modules/cache"
  
  project_name = local.project_name
  environment  = local.environment
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids
  
  # Redis cluster for high-performance caching
  engine         = "redis"
  engine_version = "7.0"
  node_type     = "cache.r6g.2xlarge"
  
  # Cluster configuration
  num_cache_clusters = 3
  parameter_group_name = "default.redis7.cluster.on"
  
  # High availability
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  # Performance settings
  parameters = [
    {
      name  = "maxmemory-policy"
      value = "allkeys-lru"
    },
    {
      name  = "notify-keyspace-events"
      value = "Ex"
    }
  ]
  
  tags = local.common_tags
}

# KMS Module for encryption
module "kms" {
  source = "./modules/kms"
  
  project_name = local.project_name
  environment  = local.environment
  
  tags = local.common_tags
}

# Monitoring Module
module "monitoring" {
  source = "./modules/monitoring"
  
  project_name = local.project_name
  environment  = local.environment
  cluster_name = local.cluster_name
  
  vpc_id            = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
  
  # CloudWatch configuration
  log_retention_days = 30
  
  # Grafana configuration
  grafana_admin_password = var.grafana_admin_password
  
  tags = local.common_tags
}

# Load Balancer Module
module "load_balancer" {
  source = "./modules/load_balancer"
  
  project_name = local.project_name
  environment  = local.environment
  
  vpc_id            = module.vpc.vpc_id
  public_subnet_ids = module.vpc.public_subnet_ids
  
  # SSL certificate
  domain_name = var.domain_name
  
  # Security
  web_acl_arn = module.waf.web_acl_arn
  
  tags = local.common_tags
}

# WAF Module for security
module "waf" {
  source = "./modules/waf"
  
  project_name = local.project_name
  environment  = local.environment
  
  # Rate limiting for API protection
  rate_limit_rules = [
    {
      name  = "api-rate-limit"
      limit = 10000  # requests per 5 minutes
      path  = "/api/*"
    },
    {
      name  = "webhook-rate-limit" 
      limit = 1000
      path  = "/webhooks/*"
    }
  ]
  
  tags = local.common_tags
}

# S3 Module for data storage
module "s3" {
  source = "./modules/s3"
  
  project_name = local.project_name
  environment  = local.environment
  
  # Data lakes for historical data
  buckets = {
    market_data = {
      versioning_enabled = true
      lifecycle_rules = [
        {
          id     = "market_data_lifecycle"
          status = "Enabled"
          transitions = [
            {
              days          = 30
              storage_class = "STANDARD_IA"
            },
            {
              days          = 90
              storage_class = "GLACIER"
            },
            {
              days          = 365
              storage_class = "DEEP_ARCHIVE"
            }
          ]
        }
      ]
    }
    
    backups = {
      versioning_enabled = true
      lifecycle_rules = [
        {
          id     = "backup_lifecycle"
          status = "Enabled"
          transitions = [
            {
              days          = 7
              storage_class = "STANDARD_IA"
            },
            {
              days          = 30
              storage_class = "GLACIER"
            }
          ]
        }
      ]
    }
    
    terraform_state = {
      versioning_enabled = true
      server_side_encryption_configuration = {
        kms_key_id = module.kms.s3_key_arn
      }
    }
  }
  
  tags = local.common_tags
}

# IAM Module
module "iam" {
  source = "./modules/iam"
  
  project_name = local.project_name
  environment  = local.environment
  cluster_name = local.cluster_name
  
  eks_cluster_arn = module.eks.cluster_arn
  
  tags = local.common_tags
}