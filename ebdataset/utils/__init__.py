from pathlib import Path, PurePath
import urllib.request
from zipfile import ZipFile
from tqdm import tqdm


def download(url, download_path, verbose=True, block_size=2048, desc="Downloading"):
    res = urllib.request.urlopen(url)
    size = int(res.info().get("Content-Length", -1))
    if size == -1:
        return False

    Path(download_path).parent.mkdir(parents=True, exist_ok=True)  # Make sure directory structure exists

    with open(download_path, "wb") as f_dst, tqdm(
        total=size, unit="B", unit_scale=True, desc=desc, disable=not verbose
    ) as pbar:
        while True:
            buf = res.read(block_size)
            if not buf:
                break
            f_dst.write(buf)
            pbar.update(len(buf))
    return True


def unzip(zip_file_path, output_directory, verbose=True, desc="Extracting"):
    with ZipFile(zip_file_path, "r") as zf:
        size = sum((f.file_size for f in zf.infolist()))
        with tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            desc=desc,
            disable=not verbose,
        ) as pbar:
            for file in zf.infolist():
                if file.is_dir():
                    continue
                path = PurePath(output_directory).joinpath(file.filename)
                Path(path.parent).mkdir(parents=True, exist_ok=True)
                zf.extract(member=file, path=path)
                pbar.update(file.file_size)
    return True
