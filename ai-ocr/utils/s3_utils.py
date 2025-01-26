import os
import boto3

class S3Client:
    
    def __init__(self):
        self.minio_url = os.getenv("MINIO_URL")
        self.access_key = os.getenv("S3_ACESS_KEY")
        self.secret_key = os.getenv("S3_SECRET_KEY")

        # Create a session using MinIO credentials
        session = boto3.session.Session()

        # Create an S3 client
        self.s3_client = session.client(
            service_name='s3',
            endpoint_url=self.minio_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
    def list_buckets(self):
        response = self.s3_client.list_buckets()
        buckets = response['Buckets']
        return buckets
    
    def download_file(self, bucket_name, s3_path, local_path):
        try:
            # Download the file from S3
            self.s3_client.download_file(bucket_name, s3_path, local_path)
            print(f'Successfully downloaded {s3_path} to {local_path}')
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    def upload_file(self, bucket_name, local_path, s3_path):
        try:
            self.s3_client.upload_file(local_path, bucket_name, s3_path)
            print(f'Successfully uploaded {local_path} to {s3_path} in bucket {bucket_name}')
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
