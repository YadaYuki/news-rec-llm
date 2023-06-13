from const.path import DATASET_DIR
import requests
from tqdm import tqdm


MIND_DATASET_BASE_URL = "https://mind201910small.blob.core.windows.net/release"
MIND_ZIP_DIR = DATASET_DIR / "mind" / "zip"


def _download_mind(zip_filename: str) -> None:
    dataset_url = f"{MIND_DATASET_BASE_URL}/{zip_filename}"
    res = requests.get(dataset_url, stream=True)
    KB = 1024
    data_size = int(res.headers.get("content-length", 0))
    with open(MIND_ZIP_DIR / zip_filename, "wb") as file:
        for chunk in tqdm(
            res.iter_content(KB, False), unit_scale=True, unit_divisor=KB, total=data_size / KB, unit="B"
        ):
            file.write(chunk)

    print(f"{zip_filename} completed.")


# training_small_url = f"{MIND_DATASET_BASE_URL}/MINDsmall_train.zip"
# validation_small_url = f'{base_url}/MINDsmall_dev.zip'
# training_large_url = f'{base_url}/MINDlarge_train.zip'
# validation_large_url = f'{base_url}/MINDlarge_dev.zip'
# test_large_url = f'{base_url}/MINDlarge_test.zip'
_download_mind("MINDsmall_train.zip")
