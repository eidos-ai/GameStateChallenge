from typing import Any
import neptune  # type: ignore
import uuid
import os


class NeptuneLogger:
    def __init__(self, enable_run: bool, project: str, run_name_or_id: str) -> None:

        self.enable_run = enable_run is not None and enable_run
        if enable_run:
            # check if neptune
            self.check_neptune_token_is_available()
            # Try if user resuming existing Experiment
            try:
                self.neptune_run = neptune.init_run(
                    project=project, with_id=run_name_or_id
                )
            except neptune.exceptions.RunNotFound as e:
                # If not found, create one
                self.neptune_run = neptune.init_run(
                    project=project, name=run_name_or_id
                )
            self.run_id = self.neptune_run.get_url().split("/")[-1]
        else:
            self.run_id = str(uuid.uuid4())

    def check_neptune_token_is_available(self) -> None:
        NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")
        if NEPTUNE_API_TOKEN is None:
            raise Exception("NEPTUNE_API_TOKEN is required")

    def get_id(self) -> Any:
        return self.run_id

    def param(self, name: str, value: Any) -> None:
        if self.enable_run:
            self.neptune_run[name] = value

    def tags(self, tags_list: list[str]) -> None:
        if self.enable_run:
            self.neptune_run["sys/tags"].add(tags_list)

    def log(self, name: str, value: Any) -> None:
        if self.enable_run:
            self.neptune_run[name].log(value)

    def log_i(self, name: str, value: Any, i: int) -> None:
        if self.enable_run:
            self.neptune_run[name].log(value, i)

    # def upload_df(self, name, df):
    #     if self.enable_run:
    #         self.neptune_run[name].upload(neptune.types.File.as_html(df))

    def upload_file(self, name: str, path: str) -> None:
        if self.enable_run:
            self.neptune_run[name].upload(str(path))

    def stop(self) -> None:
        if self.enable_run:
            self.neptune_run.stop()

    def fetch(self, param_name: str) -> Any:
        if self.enable_run:
            return self.neptune_run[param_name].fetch()
