import boto3
import pandas as pd
import numpy as np


def s3_csv_loader(
    bucket,
    prefix,
    endpoint_url="https://s3.amazonaws.com",
    delimiter=",",
    header=0,
):
    """Loads multi-axis data from csv files in an S3 bucket. 
    The csv files must be organized in subdirectories, each subdirectory 
    containing the csv files of a class. The csv files must have the same 
    headers. If not, a warning will be displayed, and the files with 
    different headers will be ignored.

    Args:
        bucket (str): Name of the S3 bucket.
        prefix (str): Prefix in the S3 bucket where the csv files are stored.
        endpoint_url (str, optional): Endpoint url of the S3 bucket.
        delimiter (str, optional): Delimiter of the csv files. Defaults to ','.
        header (int, optional): Row of the header. Defaults to 0.

    Returns:
        df: pandas Dataframe containing the data.
    """

    s3 = boto3.client("s3", endpoint_url=endpoint_url)

    warning_flag = 0
    instance_counter = 0
    headers = None  # Initialize headers

    df = pd.DataFrame(columns=["instance", "label", "feature", "data"])

    # List all objects in the bucket with the given prefix
    objects = s3.list_objects(Bucket=bucket, Prefix=prefix).get("Contents", [])

    for obj in objects:
        instance_counter += 1
        file_key = obj["Key"]
        file_obj = s3.get_object(Bucket=bucket, Key=file_key)
        file_body = file_obj["Body"]
        local_df = pd.read_csv(file_body, delimiter=delimiter, header=header)

        # Check if headers are initialized
        if headers is None:
            headers = local_df.columns.tolist()

        # Check if the headers are the same
        if local_df.columns.tolist() != headers:
            if warning_flag == 0:
                print(
                    f"Warning! Different headers detected in {file_key}. "
                    "This file and every file that does not have the same \
                    headers as "
                    f"the first file ({headers}) will be ignored. This \
                    message will be displayed only once."
                )
                warning_flag = 1
            instance_counter -= 1
            continue

        # Add csv to dataframe. 
        # Each csv column goes to a different row of the dataframe
        for i in range(len(local_df.columns)):
            data = local_df.iloc[:, i].values.astype(np.float32)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            [
                                instance_counter,
                                file_key.split("/")[
                                    1
                                ],  # Extract label from file_key
                                local_df.columns[i],
                                data,
                            ]
                        ],
                        columns=[
                            "instance",
                            "label",
                            "feature",
                            "data",
                        ],
                    ),
                ],
                ignore_index=True,
            )

    print("Loaded classes: " + str(df["label"].unique()))
    return df
