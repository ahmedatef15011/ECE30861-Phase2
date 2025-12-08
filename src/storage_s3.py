"""S3 storage implementation for model artifacts."""
import os
import io
import tempfile
import tarfile
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from typing import Optional, Tuple, List, Generator
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Configure boto3 for large file uploads
MULTIPART_THRESHOLD = 8 * 1024 * 1024  # 8MB
MULTIPART_CHUNKSIZE = 8 * 1024 * 1024  # 8MB
MAX_CONCURRENCY = 10


class S3Storage:
    """Handle S3 operations for model storage."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region: str = "us-east-1"
    ):
        """Initialize S3 client."""
        self.bucket_name = bucket_name or os.getenv(
            "S3_BUCKET_NAME",
            "mlregistry-artifacts"
        )
        self.region = region
        
        # Configure for large file transfers
        config = Config(
            max_pool_connections=MAX_CONCURRENCY,
            retries={'max_attempts': 3}
        )
        self.s3_client = boto3.client(
            's3',
            region_name=region,
            config=config
        )

        # Verify bucket exists
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Connected to S3 bucket: {self.bucket_name}")
        except ClientError as e:
            logger.error(
                f"S3 bucket {self.bucket_name} not accessible: {e}"
            )
            raise

    def stream_files_to_s3_compressed(
        self,
        file_urls: List[Tuple[str, str]],
        s3_key: str,
        artifact_name: str
    ) -> Tuple[str, int]:
        """
        Stream files directly from URLs to S3 as compressed tar.gz.
        
        Downloads files in parallel and creates a compressed archive
        that is streamed directly to S3 without storing locally.
        
        Args:
            file_urls: List of (filename, download_url) tuples
            s3_key: S3 key for the compressed archive
            artifact_name: Name for the artifact directory in archive
            
        Returns:
            Tuple of (s3_key, file_size_bytes)
        """
        try:
            # Create a temporary file for the tar.gz (streaming to S3
            # requires knowing content length for standard upload)
            with tempfile.NamedTemporaryFile(
                suffix='.tar.gz',
                delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name
                
                logger.info(
                    f"Creating compressed archive for {len(file_urls)} files"
                )
                
                # Create tar.gz archive
                with tarfile.open(tmp_path, 'w:gz') as tar:
                    # Download and add files in parallel batches
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = {}
                        for filename, url in file_urls:
                            future = executor.submit(
                                self._download_file_content,
                                url,
                                filename
                            )
                            futures[future] = filename
                        
                        for future in as_completed(futures):
                            filename = futures[future]
                            try:
                                content = future.result()
                                if content:
                                    # Add to tar
                                    info = tarfile.TarInfo(
                                        name=f"{artifact_name}/{filename}"
                                    )
                                    info.size = len(content)
                                    tar.addfile(
                                        info,
                                        io.BytesIO(content)
                                    )
                                    logger.debug(
                                        f"Added {filename} ({len(content)} bytes)"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to download {filename}: {e}"
                                )
                
                # Get final file size
                file_size = os.path.getsize(tmp_path)
                
                # Upload to S3
                logger.info(
                    f"Uploading {file_size} bytes to s3://{self.bucket_name}/{s3_key}"
                )
                
                self.s3_client.upload_file(
                    tmp_path,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={
                        'ContentType': 'application/gzip',
                        'Metadata': {
                            'artifact-name': artifact_name,
                            'compressed-size': str(file_size),
                            'file-count': str(len(file_urls))
                        }
                    }
                )
                
                logger.info(f"Upload complete: {s3_key}")
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                return s3_key, file_size
                
        except Exception as e:
            logger.error(f"Failed to stream files to S3: {e}")
            raise

    def _download_file_content(
        self,
        url: str,
        filename: str
    ) -> Optional[bytes]:
        """Download file content from URL."""
        try:
            response = requests.get(
                url,
                timeout=60,
                stream=True,
                headers={'User-Agent': 'MLRegistry/1.0'}
            )
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.warning(f"Failed to download {filename} from {url}: {e}")
            return None

    def upload_directory(
        self,
        local_path: str,
        s3_prefix: str,
        artifact_name: str
    ) -> Tuple[str, int]:
        """
        Upload a directory to S3 as a compressed tar.gz.

        Args:
            local_path: Local directory path to upload
            s3_prefix: S3 prefix (e.g., "models/1")
            artifact_name: Name for the artifact file

        Returns:
            Tuple of (s3_key, file_size_bytes)
        """
        try:
            # Create compressed archive
            with tempfile.TemporaryDirectory() as tmp_dir:
                tar_path = os.path.join(
                    tmp_dir,
                    f"{artifact_name}.tar.gz"
                )

                logger.info(
                    f"Compressing {local_path} to {tar_path}"
                )
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(local_path, arcname=artifact_name)

                # Get file size
                file_size = os.path.getsize(tar_path)

                # Upload to S3
                s3_key = f"{s3_prefix}/{artifact_name}.tar.gz"
                logger.info(
                    f"Uploading to s3://{self.bucket_name}/{s3_key}"
                )

                self.s3_client.upload_file(
                    tar_path,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={
                        'ContentType': 'application/gzip',
                        'Metadata': {
                            'artifact-name': artifact_name,
                            'compressed-size': str(file_size)
                        }
                    }
                )

                logger.info(f"Upload complete: {file_size} bytes")
                return s3_key, file_size

        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise

    def generate_download_url(
        self,
        s3_key: str,
        expiration: int = 3600
    ) -> str:
        """
        Generate a presigned URL for downloading.

        Args:
            s3_key: S3 object key
            expiration: URL expiration in seconds (default 1 hour)

        Returns:
            Presigned download URL
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise

    def delete_artifact(self, s3_key: str) -> bool:
        """Delete an artifact from S3."""
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete from S3: {e}")
            return False

    def artifact_exists(self, s3_key: str) -> bool:
        """Check if an artifact exists in S3."""
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return True
        except ClientError:
            return False


# Global instance
_s3_storage = None


def get_s3_storage() -> S3Storage:
    """Get or create S3 storage instance."""
    global _s3_storage
    if _s3_storage is None:
        _s3_storage = S3Storage()
    return _s3_storage
