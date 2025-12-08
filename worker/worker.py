"""
ECS Worker for HuggingFace Model Ingestion to S3
Polls SQS queue for ingestion messages, downloads from HuggingFace, uploads to S3
"""
import os
import sys
import json
import time
import logging
import tarfile
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import boto3
from botocore.exceptions import ClientError
from huggingface_hub import snapshot_download, HfApi
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Environment variables
SQS_QUEUE_URL = os.environ.get('SQS_QUEUE_URL')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-registery-artifacts')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
DATABASE_URL = os.environ.get('DATABASE_URL')
POLL_INTERVAL = int(os.environ.get('POLL_INTERVAL', '20'))  # Long polling interval
MAX_MESSAGES = int(os.environ.get('MAX_MESSAGES', '1'))
VISIBILITY_TIMEOUT = int(os.environ.get('VISIBILITY_TIMEOUT', '3600'))  # 1 hour

# AWS clients
sqs_client = boto3.client('sqs', region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)
hf_api = HfApi()

# Database setup (optional - for status updates)
db_engine = None
SessionLocal = None
if DATABASE_URL:
    try:
        db_engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
        logger.info(f"✅ Database connection configured: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'local'}")
    except Exception as e:
        logger.warning(f"⚠️  Database connection failed: {e}")
        logger.warning("   Worker will run without DB status updates")


def get_db() -> Optional[Session]:
    """Get database session if available."""
    if SessionLocal:
        return SessionLocal()
    return None


def update_package_status(artifact_id: str, status: str, s3_key: str = None, error: str = None):
    """Update package ingest status in database."""
    db = get_db()
    if not db:
        logger.warning(f"⚠️  Cannot update status for artifact {artifact_id} - no DB connection")
        return
    
    try:
        from src.database.models import Package
        
        package = db.query(Package).filter(Package.id == int(artifact_id)).first()
        if package:
            package.ingest_status = status
            if s3_key:
                package.s3_key = s3_key
                package.s3_bucket = S3_BUCKET_NAME
            if error:
                package.quality_gate_result = package.quality_gate_result or {}
                package.quality_gate_result['error'] = error
            package.updated_at = datetime.utcnow()
            db.commit()
            logger.info(f"✅ Updated artifact {artifact_id}: status={status}")
        else:
            logger.warning(f"⚠️  Artifact {artifact_id} not found in database")
    except Exception as e:
        logger.error(f"❌ Failed to update artifact {artifact_id}: {e}")
        db.rollback()
    finally:
        db.close()


def download_huggingface_content(
    repo_id: str,
    repo_type: str = "model",
    revision: str = "main",
    local_dir: Path = None
) -> Path:
    """
    Download content from HuggingFace.
    
    Args:
        repo_id: Repository ID (e.g., "bert-base-uncased" or "owner/model")
        repo_type: Type of repository (model, dataset, space)
        revision: Git revision (branch, tag, commit)
        local_dir: Local directory to download to
        
    Returns:
        Path to downloaded content
    """
    logger.info(f"📥 Downloading from HuggingFace: {repo_type}/{repo_id}@{revision}")
    
    try:
        # Create temp directory if not provided
        if local_dir is None:
            local_dir = Path(tempfile.mkdtemp(prefix=f"hf_{repo_id.replace('/', '_')}_"))
        
        # Download using snapshot_download (handles large files with git-lfs)
        download_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # Don't use symlinks for S3 upload
            resume_download=True,  # Resume if interrupted
        )
        
        logger.info(f"✅ Downloaded to: {download_path}")
        return Path(download_path)
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise


def create_tarball(source_dir: Path, output_path: Path = None) -> Path:
    """
    Create a compressed tarball from directory.
    
    Args:
        source_dir: Directory to compress
        output_path: Output tar.gz file path
        
    Returns:
        Path to created tarball
    """
    if output_path is None:
        output_path = source_dir.parent / f"{source_dir.name}.tar.gz"
    
    logger.info(f"📦 Creating tarball: {output_path.name}")
    
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Tarball created: {file_size_mb:.2f} MB")
    
    return output_path


def upload_to_s3(
    file_path: Path,
    s3_key: str,
    bucket: str = S3_BUCKET_NAME,
    metadata: Dict[str, str] = None
) -> str:
    """
    Upload file to S3 with multipart upload for large files.
    
    Args:
        file_path: Local file to upload
        s3_key: S3 object key
        bucket: S3 bucket name
        metadata: Optional metadata to attach
        
    Returns:
        S3 URL of uploaded object
    """
    logger.info(f"☁️  Uploading to S3: s3://{bucket}/{s3_key}")
    
    # Check if file already exists in S3
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_key)
        s3_url = f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"✅ Already exists in S3, skipping upload: {s3_url}")
        return s3_url
    except s3_client.exceptions.NoSuchKey:
        # File doesn't exist, proceed with upload
        pass
    except Exception as e:
        # Other error, log but continue with upload attempt
        logger.warning(f"⚠️  Error checking S3 existence: {e}")
    
    try:
        file_size = file_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        # Use multipart upload for files > 100MB
        if file_size > 100 * 1024 * 1024:
            logger.info(f"   Using multipart upload ({file_size_mb:.2f} MB)")
            upload_multipart(file_path, s3_key, bucket, metadata)
        else:
            logger.info(f"   Using simple upload ({file_size_mb:.2f} MB)")
            extra_args = {'Metadata': metadata} if metadata else {}
            s3_client.upload_file(str(file_path), bucket, s3_key, ExtraArgs=extra_args)
        
        s3_url = f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"✅ Uploaded successfully: {s3_url}")
        
        return s3_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise


def upload_multipart(
    file_path: Path,
    s3_key: str,
    bucket: str,
    metadata: Dict[str, str] = None,
    chunk_size: int = 8 * 1024 * 1024  # 8MB chunks
):
    """Upload large file to S3 using multipart upload."""
    try:
        # Initiate multipart upload
        mpu_params = {}
        if metadata:
            mpu_params['Metadata'] = metadata
            
        mpu = s3_client.create_multipart_upload(
            Bucket=bucket,
            Key=s3_key,
            **mpu_params
        )
        upload_id = mpu['UploadId']
        
        # Upload parts
        parts = []
        part_number = 1
        
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                
                logger.info(f"   Uploading part {part_number}...")
                
                response = s3_client.upload_part(
                    Bucket=bucket,
                    Key=s3_key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=data
                )
                
                parts.append({
                    'ETag': response['ETag'],
                    'PartNumber': part_number
                })
                
                part_number += 1
        
        # Complete multipart upload
        s3_client.complete_multipart_upload(
            Bucket=bucket,
            Key=s3_key,
            UploadId=upload_id,
            MultipartUpload={'Parts': parts}
        )
        
        logger.info(f"   Multipart upload complete: {part_number - 1} parts")
        
    except Exception as e:
        # Abort multipart upload on error
        try:
            s3_client.abort_multipart_upload(
                Bucket=bucket,
                Key=s3_key,
                UploadId=upload_id
            )
        except:
            pass
        raise


def create_metadata_json(
    artifact_id: str,
    repo_id: str,
    repo_type: str,
    source_url: str,
    s3_key: str,
    status: str = "completed"
) -> Dict[str, Any]:
    """Create metadata JSON for uploaded artifact."""
    return {
        "artifact_id": artifact_id,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "source_url": source_url,
        "s3_key": s3_key,
        "s3_bucket": S3_BUCKET_NAME,
        "s3_region": AWS_REGION,
        "status": status,
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "worker_version": "1.0.0"
    }


def process_message(message: Dict[str, Any]) -> bool:
    """
    Process a single SQS message.
    
    Args:
        message: SQS message dictionary
        
    Returns:
        True if processing succeeded, False otherwise
    """
    receipt_handle = message['ReceiptHandle']
    
    try:
        # Parse message body
        body = json.loads(message['Body'])
        
        artifact_id = body['artifact_id']
        repo_id = body['repo_id']
        repo_type = body.get('repo_type', 'model')
        revision = body.get('revision', 'main')
        source_url = body['source_url']
        artifact_type = body.get('artifact_type', 'model')
        requested_at = body.get('requested_at', 'unknown')
        
        logger.info("=" * 70)
        logger.info(f"🔨 PROCESSING MESSAGE")
        logger.info(f"   Artifact ID: {artifact_id}")
        logger.info(f"   Repo: {repo_type}/{repo_id}")
        logger.info(f"   Source: {source_url}")
        logger.info(f"   Requested: {requested_at}")
        logger.info("=" * 70)
        
        # Update status to processing
        update_package_status(artifact_id, "processing")
        
        # Step 1: Download from HuggingFace
        temp_dir = Path(tempfile.mkdtemp(prefix=f"worker_{artifact_id}_"))
        download_dir = temp_dir / "content"
        download_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"📥 Step 1/4: Downloading from HuggingFace...")
            content_path = download_huggingface_content(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                local_dir=download_dir
            )
            
            # Step 2: Create tarball
            logger.info(f"📦 Step 2/4: Creating tarball...")
            tarball_path = create_tarball(content_path, temp_dir / f"{artifact_id}.tar.gz")
            
            # Step 3: Upload to S3
            logger.info(f"☁️  Step 3/4: Uploading to S3...")
            s3_key = f"{artifact_type}s/{artifact_id}/{artifact_id}.tar.gz"
            
            metadata = {
                "artifact-id": str(artifact_id),
                "repo-id": str(repo_id),
                "repo-type": str(repo_type),
                "source-url": str(source_url),
                "processed-at": str(datetime.utcnow().isoformat())
            }
            
            s3_url = upload_to_s3(tarball_path, s3_key, S3_BUCKET_NAME, metadata)
            
            # Step 4: Upload metadata JSON
            logger.info(f"📝 Step 4/4: Uploading metadata...")
            metadata_json = create_metadata_json(
                artifact_id, repo_id, repo_type, source_url, s3_key, "completed"
            )
            metadata_key = f"{artifact_type}s/{artifact_id}/metadata.json"
            
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=metadata_key,
                Body=json.dumps(metadata_json, indent=2),
                ContentType='application/json'
            )
            
            # Update database status
            update_package_status(artifact_id, "completed", s3_key)
            
            logger.info("=" * 70)
            logger.info(f"✅ PROCESSING COMPLETE")
            logger.info(f"   S3 Key: {s3_key}")
            logger.info(f"   S3 URL: {s3_url}")
            logger.info("=" * 70)
            
            # Delete message from queue
            sqs_client.delete_message(
                QueueUrl=SQS_QUEUE_URL,
                ReceiptHandle=receipt_handle
            )
            logger.info(f"🗑️  Message deleted from queue")
            
            return True
            
        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"🧹 Cleaned up temp directory")
        
    except Exception as e:
        logger.error(f"❌ PROCESSING FAILED: {e}", exc_info=True)
        
        # Update status to failed
        try:
            body = json.loads(message['Body'])
            artifact_id = body.get('artifact_id', 'unknown')
            update_package_status(artifact_id, "failed", error=str(e))
        except:
            pass
        
        # Don't delete message - let it retry or go to DLQ
        return False


def poll_messages():
    """Poll SQS queue for messages and process them."""
    logger.info("🚀 Worker started")
    logger.info(f"   Queue: {SQS_QUEUE_URL}")
    logger.info(f"   Bucket: {S3_BUCKET_NAME}")
    logger.info(f"   Region: {AWS_REGION}")
    logger.info(f"   Poll Interval: {POLL_INTERVAL}s")
    logger.info(f"   Database: {'Connected' if SessionLocal else 'Not configured'}")
    logger.info("=" * 70)
    
    while True:
        try:
            logger.info("📬 Polling for messages...")
            
            # Long polling (wait up to POLL_INTERVAL seconds)
            response = sqs_client.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=MAX_MESSAGES,
                WaitTimeSeconds=POLL_INTERVAL,
                VisibilityTimeout=VISIBILITY_TIMEOUT,
                MessageAttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            
            if not messages:
                logger.info("   No messages available")
                continue
            
            logger.info(f"📨 Received {len(messages)} message(s)")
            
            # Process each message
            for message in messages:
                try:
                    process_message(message)
                except Exception as e:
                    logger.error(f"❌ Message processing error: {e}", exc_info=True)
            
        except KeyboardInterrupt:
            logger.info("🛑 Worker shutting down (KeyboardInterrupt)")
            break
        except Exception as e:
            logger.error(f"❌ Polling error: {e}", exc_info=True)
            logger.info(f"   Retrying in 10 seconds...")
            time.sleep(10)


def main():
    """Main entry point."""
    # Validate environment variables
    if not SQS_QUEUE_URL:
        logger.error("❌ SQS_QUEUE_URL environment variable is required")
        sys.exit(1)
    
    if not S3_BUCKET_NAME:
        logger.error("❌ S3_BUCKET_NAME environment variable is required")
        sys.exit(1)
    
    # Start polling
    poll_messages()


if __name__ == "__main__":
    main()
