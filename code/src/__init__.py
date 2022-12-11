import os


def _chdir_to_project_path() -> str:
    path: str = os.path.realpath(__file__)
    path = path.replace("\\", "/")
    path = "/".join(path.split("/")[:-3])
    os.chdir(path)


PATH_DATA = "data/LJSpeech-1.1"
