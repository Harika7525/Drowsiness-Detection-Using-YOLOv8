import sys
from Drowsiness_Detection.component import data_ingestion
from Drowsiness_Detection.logger import logging
from Drowsiness_Detection.exception import AppException
from Drowsiness_Detection.component.data_ingestion import DataIngestion

from Drowsiness_Detection.entity.config_entity import DataIngestionConfig
from Drowsiness_Detection.entity.artifacts_entity import DataIngestionArtifact


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_Data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class"
            )
            logging.info("Getting the data from URL")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from URL")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )

            return data_ingestion_artifact
        except Exception as e:
            raise AppException(e, sys)

    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_Data_ingestion()

        except Exception as e:
            raise AppException(e, sys)
