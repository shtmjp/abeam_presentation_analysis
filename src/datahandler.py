"""DataHandler module.

Copyright (C) 2024 Takaaki Shiotani

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pickle
from pathlib import Path
from typing import Literal

import ffmpeg
from yaml import safe_load

from presentation import Presentation


class DataHandler:
    """DataHandler class."""

    PROJECT_ROOT = Path(__file__).parent.parent
    METADATA_PATH = PROJECT_ROOT / "data" / "metadata.yaml"
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "abeam_raw"
    DATA_DIR = PROJECT_ROOT / "data" / "abeam_presentation"

    def __init__(self) -> None:
        """Initialize DataHandler."""
        # load metadata yaml
        with self.METADATA_PATH.open("r") as file:
            data = safe_load(file)
        self.names: list[str] = list(data.keys())
        self.name2indexes: dict[str, list[int]] = {
            name: d["indexes"] for name, d in data.items()
        }
        self.name2ispro: dict[str, list[bool]] = {
            name: d["is_proficient"] for name, d in data.items()
        }
        self.name2label: dict[str, int] = {name: d["label"] for name, d in data.items()}
        self.name2sex: dict[str, Literal["male", "female"]] = {
            name: d["sex"] for name, d in data.items()
        }

    def get_raw_mp4_path(self, name: str, index: int) -> Path:
        """Get raw mp4 path for given name and index."""
        return self.RAW_DATA_DIR / name / f"{name}{index}.mp4"

    def out_wav(self, name: str, index: int) -> None:
        """Obtain and save wav file from mp4 file for given name and index."""
        in_file_str = str(self.get_raw_mp4_path(name, index))
        out_dir = self.DATA_DIR / name
        out_dir.mkdir(exist_ok=True, parents=True)
        out_file_str = str(out_dir / f"{name}{index}.wav")
        stream = ffmpeg.input(in_file_str)
        stream = ffmpeg.output(stream, out_file_str)
        ffmpeg.run(stream)

    def out_wav_all(self) -> None:
        """Obtain and save wav files from all presentations."""
        for name in self.names:
            for index in self.name2indexes[name]:
                if self.get_wav_path(name, index).exists():
                    continue
                self.out_wav(name, index)

    def get_wav_path(self, name: str, index: int) -> Path:
        """Get wav path for given name and index."""
        return self.DATA_DIR / name / f"{name}{index}.wav"

    def get_pickle_path(self, name: str, index: int) -> Path:
        """Get pickle path of the Presentation object for given name and index."""
        return self.DATA_DIR / name / "processed" / f"{name}{index}.pkl"

    def load_presentation(self, name: str, index: int) -> Presentation:
        """Load Presentation object for given name and index."""
        with self.get_pickle_path(name, index).open("rb") as f:
            p = pickle.load(f)  # noqa: S301
        if p.name != name:
            msg = f"Expected name {name}, but got {p.name}"
            raise AssertionError(msg)
        if p.number != index:
            msg = f"Expected index {index}, but got {p.number}"
            raise AssertionError(msg)
        return p

    def save_presentation(
        self,
        p: Presentation,
        is_force: bool = False,  # noqa: FBT001, FBT002
        save_dir: Path = DATA_DIR,
    ) -> None:
        """Save Presentation object."""
        if not is_force and self.get_pickle_path(p.name, p.number).exists():
            msg = "Pickle file already exists"
            raise AssertionError(msg)
        save_dir = self.get_pickle_path(p.name, p.number).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        with self.get_pickle_path(p.name, p.number).open("wb") as f:
            pickle.dump(p, f)
