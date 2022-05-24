# configuration parameters
from pathlib import Path

from boto3.s3.transfer import TransferConfig

AWS_REGION = 'us-east-1'
AWS_IAM_SECRETS_FILE = Path(
    Path(__file__).parent, "secrets", "s3_iam_secrets.csv"
)
BUCKET_NAME = "bucket name"
# note: these are quite conservative settings
TRANSFERCONFIG = TransferConfig(
    max_concurrency=2, multipart_chunksize=32*1024**2
)
