from typing import Any
import random

import pyarrow as pa
from data_processing.utils import MB, KB, get_logger

logger = get_logger(__name__)


class DataAccess:
    """
    Base class for data access (interface), defining all the methods
    """

    def get_files_to_process(self) -> tuple[list[str], dict[str, float]]:
        """
        Get files to process
        :return: list of files and a dictionary of the files profile:
            "max_file_size_MB",
            "min_file_size_MB",
            "avg_file_size_MB",
            "total_file_size_MB"
        """
        pass

    def get_table(self, path: str) -> pa.table:
        """
        Get pyArrow table for a given path
        :param path - file path
        :return: pyArrow table or None, if the table read failed
        """
        pass

    def get_file(self, path: str) -> bytes:
        """
        Get file as a byte array
        :param path: file path
        :return: bytes array of file content
        """
        pass

    def get_folder_files(self, path: str, extensions: list[str] = None, return_data: bool = True) -> dict[str, bytes]:
        """
        Get a list of byte content of files. The path here is an absolute path and can be anywhere.
        The current limitation for S3 and Lakehouse is that it has to be in the same bucket
        :param path: file path
        :param extensions: a list of file extensions to include. If None, then all files from this and
                           child ones will be returned
        :param return_data: flag specifying whether the actual content of files is returned (True), or just
                            directory is returned (False)
        :return: A dictionary of file names/binary content will be returned
        """
        pass

    def save_file(self, path: str, data: bytes) -> dict[str, Any]:
        """
        Save byte array to the file
        :param path: file path
        :param data: byte array
        :return: a dictionary as
        defined https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/put_object.html
        in the case of failure dict is None
        """

    def get_output_location(self, path: str) -> str:
        """
        Get output location based on input
        :param path: input file location
        :return: output file location
        """
        return ""

    def save_table(self, path: str, table: pa.Table) -> tuple[int, dict[str, Any]]:
        """
        Save table to a given location
        :param path: location to save table
        :param table: table
        :return: size of table in memory and a dictionary as
        defined https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/put_object.html
        in the case of failure dict is None
        """
        pass

    def save_job_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Save job metadata
        :param metadata: a dictionary, containing the following keys
        (see https://github.ibm.com/arc/dmf-library/issues/158):
            "pipeline",
            "job details",
            "code",
            "job_input_params",
            "execution_stats",
            "job_output_stats"
        two additional elements:
            "source"
            "target"
        are filled bu implementation
        :return: a dictionary as
        defined https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/put_object.html
        in the case of failure dict is None
        """
        pass

    def sample_input_data(self, n_samples: int = 10) -> dict[str, Any]:
        """
        Sample input data set to get average table size, average doc size, number of docs, etc.
        Note that here we are not reading all of the input documents, but rather randomly pick
        their subset. It gives more precise answer as subset grows, but it takes longer
        :param n_samples: number of samples to use - default 10
        :return: a dictionary of the files profile:
            "max_file_size_MB",
            "min_file_size_MB",
            "avg_file_size_MB",
            "total_file_size_MB"
            average table size MB,
            average doc size KB,
            estimated number of docs
        """
        # get files to process
        path_list, path_profile = self.get_files_to_process()
        # Pick files to sample
        if len(path_list) > n_samples:
            # Pick files at random
            files = [int(random.random() * len(path_list)) for _ in range(n_samples)]
        else:
            # use all existing files
            files = range(len(path_list))
        logger.info(f"Using files {files} to sample data")

        # Read table and compute number of docs and sizes
        number_of_docs = []
        table_sizes = []
        n_tables = 0
        for f in files:
            f_name = path_list[f]
            table = self.get_table(path=f_name)
            if table is not None:
                n_tables += 1
                number_of_docs.append(table.num_rows)
                # As a table size is mostly document, we can consider them roughly the same
                table_sizes.append(table.nbytes)
        # compute averages
        if n_tables == 0:
            av_number_docs = 0
            av_table_size = 0
            av_doc_size = 0
        else:
            av_number_docs = sum(number_of_docs) / n_tables
            av_table_size = sum(table_sizes) / n_tables / MB
            if av_number_docs == 0:
                av_doc_size = 0
            else:
                av_doc_size = av_table_size * MB / av_number_docs / KB
        logger.info(
            f"average number of docs {av_number_docs}, average table size {av_table_size} MB, "
            f"average doc size {av_doc_size} kB"
        )

        # compute number of docs
        number_of_docs = av_number_docs * len(path_list)
        logger.info(f"Estimated number of docs {number_of_docs}")
        return path_profile | {
            "average table size MB": av_table_size,
            "average doc size KB": av_doc_size,
            "estimated number of docs": number_of_docs,
        }
