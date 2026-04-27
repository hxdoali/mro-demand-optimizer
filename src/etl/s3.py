"""
AWS S3 integration for data ingestion and artifact storage.

Supports reading/writing parquet data and model artifacts to S3,
with local filesystem fallback for development.
"""

import os
import logging
from pathlib import Path
from io import BytesIO

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


S3_BUCKET = os.getenv("MRO_S3_BUCKET", "mro-demand-optimizer")
S3_PREFIX = os.getenv("MRO_S3_PREFIX", "data/")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def get_s3_client():
    """Create an S3 client with configured region."""
    if not BOTO3_AVAILABLE:
        raise ImportError("boto3 is not installed. Install with: pip install boto3")
    return boto3.client("s3", region_name=AWS_REGION)


def upload_parquet_to_s3(
    df: pd.DataFrame,
    s3_key: str,
    bucket: str = S3_BUCKET,
) -> str:
    """Upload a DataFrame as parquet to S3."""
    s3 = get_s3_client()
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    full_key = f"{S3_PREFIX}{s3_key}"
    s3.put_object(Bucket=bucket, Key=full_key, Body=buffer.getvalue())
    uri = f"s3://{bucket}/{full_key}"
    logger.info(f"Uploaded {len(df)} rows to {uri}")
    return uri


def download_parquet_from_s3(
    s3_key: str,
    bucket: str = S3_BUCKET,
) -> pd.DataFrame:
    """Download a parquet file from S3 and return as DataFrame."""
    s3 = get_s3_client()
    full_key = f"{S3_PREFIX}{s3_key}"

    response = s3.get_object(Bucket=bucket, Key=full_key)
    buffer = BytesIO(response["Body"].read())
    return pd.read_parquet(buffer)


def upload_model_artifact(
    artifact_bytes: bytes,
    artifact_name: str,
    bucket: str = S3_BUCKET,
) -> str:
    """Upload a serialized model artifact to S3."""
    s3 = get_s3_client()
    key = f"{S3_PREFIX}models/{artifact_name}"
    s3.put_object(Bucket=bucket, Key=key, Body=artifact_bytes)
    uri = f"s3://{bucket}/{key}"
    logger.info(f"Uploaded model artifact to {uri}")
    return uri


def list_s3_objects(prefix: str = "", bucket: str = S3_BUCKET) -> list[dict]:
    """List objects in the S3 bucket under a given prefix."""
    s3 = get_s3_client()
    full_prefix = f"{S3_PREFIX}{prefix}"

    response = s3.list_objects_v2(Bucket=bucket, Prefix=full_prefix)
    objects = []
    for obj in response.get("Contents", []):
        objects.append({
            "key": obj["Key"],
            "size_mb": round(obj["Size"] / (1024 * 1024), 2),
            "last_modified": obj["LastModified"].isoformat(),
        })
    return objects


def sync_local_to_s3(
    local_dir: str = "data/synthetic",
    bucket: str = S3_BUCKET,
) -> list[str]:
    """Upload all parquet files from a local directory to S3."""
    uploaded = []
    for path in Path(local_dir).glob("*.parquet"):
        df = pd.read_parquet(path)
        uri = upload_parquet_to_s3(df, path.name, bucket)
        uploaded.append(uri)
    return uploaded


def load_from_s3_or_local(
    s3_key: str,
    local_path: str,
    bucket: str = S3_BUCKET,
) -> pd.DataFrame:
    """
    Try loading from S3 first, fall back to local file.
    Enables seamless dev (local) vs prod (S3) workflows.
    """
    if BOTO3_AVAILABLE and os.getenv("USE_S3", "false").lower() == "true":
        try:
            logger.info(f"Loading from S3: s3://{bucket}/{S3_PREFIX}{s3_key}")
            return download_parquet_from_s3(s3_key, bucket)
        except (ClientError, NoCredentialsError) as e:
            logger.warning(f"S3 load failed ({e}), falling back to local")

    logger.info(f"Loading from local: {local_path}")
    return pd.read_parquet(local_path)
