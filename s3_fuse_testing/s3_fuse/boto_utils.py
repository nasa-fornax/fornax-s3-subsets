"""
convenience wrappers for transferring objects via boto to reduce
'contamination' of other components
"""
import io
from functools import partial
from pathlib import Path
from typing import Callable, Union, Optional, IO

import boto3
from boto3.s3.transfer import MB, TransferConfig
import botocore.client
import botocore.config
from botocore.exceptions import ClientError

# configuration parameters

AWS_REGION = 'us-east-1'
AWS_IAM_SECRETS_FILE = Path(
    Path(__file__).parent, "secrets", "s3_iam_secrets.csv"
)
# note: these are quite conservative settings
TRANSFERCONFIG = TransferConfig(max_concurrency=2, multipart_chunksize=32*MB)


def upload_s3(
    bucket: str,
    upload_object: Union[str, Path, IO] = None,
    object_key: Optional[str] = None,
    client: Optional[botocore.client.BaseClient] = None,
    pass_string: bool = False,
    config: Optional[TransferConfig] = None
):
    """
    Upload a file or buffer to an S3 bucket.

    :param upload_object: String, pathlike, or filelike object to upload
    :param bucket: name of bucket to upload it to
    :param object_key: S3 object name, including fully-qualified prefixes.
        If not specified then str(file_or_buffer) is used -- will most likely
        look bad if it's a buffer
    :param client: botocore.client.S3 instance; makes a default client if None
    :param pass_string -- write passed string directly to file instead of
        interpreting as a path
    :param config: optional boto3.s3.TransferConfig
    :return: None
    """
    if client is None:
        client = boto3.client("s3")
    # If S3 object_name was not specified, use string rep of
    # passed object, up to 90 characters
    if object_key is None:
        object_key = str(upload_object)[:90]
    # 'touch' - type behavior
    if upload_object is None:
        upload_object = io.BytesIO()
    # encode string to bytes if we're writing it to S3 object instead
    # of interpreting it as a path
    if isinstance(upload_object, str) and pass_string:
        upload_object = io.BytesIO(upload_object.encode("utf-8"))
    # Upload the file
    if isinstance(upload_object, (Path, str)):
        client.upload_file(
            str(upload_object), bucket, object_key, Config=config
        )
    else:
        client.upload_fileobj(
            upload_object, bucket, object_key, Config=config
        )


def download_s3(
    bucket: str,
    object_key: str,
    target: Optional[Union[str, Path, io.BytesIO]] = None,
    client: Optional[botocore.client.BaseClient] = None,
    config: Optional[TransferConfig] = None
) -> Union[ClientError, str, Path, io.BytesIO]:
    """
    Download an object from an s3 bucket to a file or buffer.

    :param bucket: name of bucket to upload it to
    :param object_key: S3 object name, including fully-qualified prefixes.
    :param target: where to put the contents of the downloaded object. if a
      str or Path, writes into a file at that path. If a buffer, writes into
      the buffer.
    :param client: botocore.client.S3 instance; makes a default client if None
    :param config: optional boto3.s3.TransferConfig

    :return: a buffer containing the file contents, or a path to the target
        file, if it was
    """
    if client is None:
        client = boto3.client("s3")
    if target is None:
        target = io.BytesIO()
    try:
        if isinstance(target, (Path, str)):
            # if passed a path-ish thing, download to that path
            client.download_file(bucket, object_key, target, Config=config)
        else:
            # otherwise, download into a buffer
            client.download_fileobj(bucket, object_key, target, Config=config)
            target.seek(0)
    except ClientError as error:
        return error
    return target


def make_default_s3_client() -> botocore.client.BaseClient:
    """convenience constructor for a boto3 s3 client."""
    aws_config = botocore.config.Config(region_name=AWS_REGION)
    with open(AWS_IAM_SECRETS_FILE) as secrets_file:
        secrets = secrets_file.readlines()[1].split(",")
    return boto3.client(
        "s3",
        aws_access_key_id=secrets[2],
        aws_secret_access_key=secrets[3],
        config=aws_config,
    )


def bind_s3_bucket(bucket: str) -> tuple[Callable, Callable]:
    """convenience constructor for s3 r/w callables backed by a boto client."""
    client = make_default_s3_client()
    return(
        partial(upload_s3, bucket, client=client, config=TRANSFERCONFIG),
        partial(download_s3, bucket, client=client, config=TRANSFERCONFIG)
    )
