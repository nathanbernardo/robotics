from typing import Optional, List
from minio import Minio
from minio.error import S3Error
import os


class MinioClientError(Exception):
    pass


class MinioClient:
    def __init__(
        self, endpoint: str, access_key: str, secret_key: str, secure: bool = False
    ):
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    def ensure_bucket(self, bucket_name: str):
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
        except S3Error as e:
            print(f"Error initializing MinIO client: {e}")
            return None

    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: Optional[str] = None,
    ) -> None:
        if not bucket_name or not object_name or not file_path:
            raise ValueError("bucket_name, object_name, and file_path cannot be empty")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self.ensure_bucket(bucket_name)

        try:
            self.client.fput_object(
                bucket_name,
                object_name,
                file_path,
                content_type=content_type or "application/octet-stream",
            )
        except S3Error as e:
            raise MinioClientError(
                f"Failed to ensure bucket '{bucket_name}': {e}"
            ) from e

    def download_file(self, bucket_name: str, object_name: str, dest_path: str):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        self.client.fget_object(bucket_name, object_name, dest_path)

    def list_objects(
        self, bucket_name: str, prefix: str = "", recursive: bool = True
    ) -> List[str | None]:
        return [
            obj.object_name
            for obj in self.client.list_objects(
                bucket_name, prefix=prefix, recursive=recursive
            )
        ]

    def get_object(self, bucket_name: str, object_name: str):
        return self.client.get_object(bucket_name, object_name)

    def get_metadata(self, bucket_name: str, object_name: str):
        return self.client.stat_object(bucket_name, object_name)
