from typing import Optional, List
from minio import Minio
from minio.error import S3Error
import os


class MinioClient:
    def __init__(
        self, endpoint: str, access_key: str, secret_key: str, secure: bool = False
    ):
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )

    def ensure_bucket(self, bucket_name: str):
        if not self.client.bucket_exists(bucket_name):
            try:
                self.client.make_bucket(bucket_name)
            except Exception as e:
                print(f"Error initializing MinIO client: {e}")
                return None

    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: Optional[str] = None,
    ):
        self.ensure_bucket(bucket_name)
        self.client.fput_object(
            bucket_name,
            object_name,
            file_path,
            content_type=content_type or "application/octet-stream",
        )

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
