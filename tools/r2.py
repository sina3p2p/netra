#!/usr/bin/env python3
"""
Cloudflare R2 storage utility for uploading and downloading files.

Usage:
    # Upload a file
    python tools/r2.py upload checkpoints/step_1000.pt

    # Upload with custom remote key
    python tools/r2.py upload checkpoints/step_1000.pt --key models/v1/step_1000.pt

    # Download a file
    python tools/r2.py download checkpoints/step_1000.pt

    # Download to a specific local path
    python tools/r2.py download checkpoints/step_1000.pt --out ./restored.pt

    # List files in bucket
    python tools/r2.py ls
    python tools/r2.py ls --prefix checkpoints/

Required env vars:
    R2_ENDPOINT_URL      - https://<account-id>.r2.cloudflarestorage.com
    R2_ACCESS_KEY_ID     - API token access key
    R2_SECRET_ACCESS_KEY - API token secret key
    R2_BUCKET            - Bucket name (or pass --bucket)
"""

import argparse
import os
import sys


def _get_client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )


def is_configured():
    """Check if R2 env vars are set."""
    return all(os.environ.get(k) for k in [
        "R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
    ]) and os.environ.get("R2_BUCKET")


def check_connection():
    """Verify R2 connection and bucket access. Returns True if ok, False otherwise."""
    if not is_configured():
        return False
    bucket = os.environ["R2_BUCKET"]
    try:
        s3 = _get_client()
        s3.head_bucket(Bucket=bucket)
        print(f"R2: connected → {bucket}")
        return True
    except Exception as e:
        print(f"R2: connection failed → {e}")
        return False


def upload(local_path, bucket, key=None):
    """Upload a local file to R2. Returns the remote key."""
    s3 = _get_client()
    if key is None:
        key = os.path.basename(local_path)
    s3.upload_file(str(local_path), bucket, key)
    return key


def download(key, bucket, local_path=None):
    """Download a file from R2. Returns the local path."""
    s3 = _get_client()
    if local_path is None:
        local_path = os.path.basename(key)
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    s3.download_file(bucket, key, str(local_path))
    return local_path


def list_files(bucket, prefix=""):
    """List files in the bucket. Returns list of (key, size_bytes) tuples."""
    s3 = _get_client()
    result = []
    kwargs = {"Bucket": bucket}
    if prefix:
        kwargs["Prefix"] = prefix
    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            result.append((obj["Key"], obj["Size"]))
        if not resp.get("IsTruncated"):
            break
        kwargs["ContinuationToken"] = resp["NextContinuationToken"]
    return result


def main():
    p = argparse.ArgumentParser(description="Cloudflare R2 file utility")
    p.add_argument("--bucket", type=str, default=os.environ.get("R2_BUCKET"))
    sub = p.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="Upload a file to R2")
    up.add_argument("file", help="Local file path")
    up.add_argument("--key", type=str, default=None, help="Remote key (default: filename)")

    dl = sub.add_parser("download", help="Download a file from R2")
    dl.add_argument("key", help="Remote key in bucket")
    dl.add_argument("--out", type=str, default=None, help="Local output path")

    ls = sub.add_parser("ls", help="List files in bucket")
    ls.add_argument("--prefix", type=str, default="", help="Filter by prefix")

    sub.add_parser("status", help="Check R2 connection")

    args = p.parse_args()

    if args.command == "status":
        ok = check_connection()
        sys.exit(0 if ok else 1)

    if not args.bucket:
        print("Error: set R2_BUCKET env var or pass --bucket", file=sys.stderr)
        sys.exit(1)

    if args.command == "upload":
        key = upload(args.file, args.bucket, key=args.key)
        print(f"Uploaded → r2://{args.bucket}/{key}")

    elif args.command == "download":
        path = download(args.key, args.bucket, local_path=args.out)
        print(f"Downloaded → {path}")

    elif args.command == "ls":
        files = list_files(args.bucket, prefix=args.prefix)
        if not files:
            print("(empty)")
        for key, size in files:
            if size > 1024**3:
                size_str = f"{size / 1024**3:.1f} GB"
            elif size > 1024**2:
                size_str = f"{size / 1024**2:.1f} MB"
            else:
                size_str = f"{size / 1024:.1f} KB"
            print(f"  {size_str:>10s}  {key}")


if __name__ == "__main__":
    main()
