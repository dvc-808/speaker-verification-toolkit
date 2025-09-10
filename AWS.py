import boto3
from botocore.exceptions import ClientError

class AWS():
    def __init__(self):
        # Create a Secrets Manager client
        session = boto3.session.Session()
        self.client = session.client(
            service_name='secretsmanager',
            region_name="ap-southeast-1"
        )

    def get_secret(self, secname):
        try:
            get_secret_value_response = self.client.get_secret_value(
                SecretId=secname
            )
        except ClientError as e:
            # For a list of exceptions thrown, see
            # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
            raise e
        secret = get_secret_value_response['SecretString']
        return secret
    # def put_s3_object():

