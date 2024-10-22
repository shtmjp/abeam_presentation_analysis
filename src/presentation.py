from __future__ import annotations

from typing import TYPE_CHECKING

import librosa
import numpy as np
import pandas as pd
import parselmouth
import stable_whisper
import torch
from fugashi import Tagger  # type: ignore[attr-defined]
from stable_whisper import WhisperResult

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path

    from torch.jit import ScriptModule
    from whisper import Whisper


def fugashi_verbose_to_df(s: str) -> pd.DataFrame:
    """Convert fugashi verbose output to DataFrame."""
    # description of columns can be found in lib/site-packages/unidic/dicdir/dicrc
    tagger = Tagger("-Overbose")
    f = tagger.parse(s)
    dict_lines = []
    for line in f.splitlines()[:-1]:  # avoid the last EOS line
        d = {}
        for feat_str in line.split("\t"):
            if feat_str == "":
                continue
            feat_name, value = feat_str.split(":")
            d[feat_name] = value
        dict_lines.append(d)
    return pd.DataFrame(dict_lines)


def get_yomi(txt: str) -> str:
    """Get yomi from text."""
    fuga_df = fugashi_verbose_to_df(txt)
    try:
        prons = fuga_df["pron"].fillna(fuga_df["surface"]).to_list()
    except KeyError:
        prons = fuga_df["surface"].to_list()
    return "".join(prons)


class ModelContainer:
    """A container class for managing large models."""

    def __init__(self) -> None:
        """Initialize the ModelContainer instance."""
        self.silero_model: ScriptModule | None = None
        self.silero_utils: list = []
        self.whisper_model: Whisper | None = None

    def load_whisper(self, name: str = "large") -> None:
        """Load the Whisper model.

        Parameters
        ----------
        name : str, optional
            Name of the Whisper model to load, by default "large".

        """
        self.whisper_model = stable_whisper.load_model(name)

    def load_silero(self) -> None:
        """Load the Silero VAD model and utilities."""
        silero_model, silero_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
        )  # type: ignore  # noqa: PGH003
        self.silero_model = silero_model
        self.silero_utils = silero_utils


class Presentation:
    """A class to represent a presentation with various audio processing capabilities.

    Attributes
    ----------
    wav_path : Path
        Path to the WAV file.
    name : str
        Name of the presentation.
    number : int
        Number identifier for the presentation.
    sample_rate : int
        Sample rate of the audio file.
    logger : Logger, optional
        Logger for logging information.
    is_preprocessed : bool
        Flag to indicate if preprocessing is done.
    active_intervals : list of tuples
        List of active intervals in the audio.
    formants_df : pd.DataFrame
        DataFrame containing formant information.
    whisper_result : object
        Result of the whisper transcription.
    yomi : str
        Yomi transcription of the text.
    pyin_result : dict
        Result of the PYIN pitch extraction.
    feature_dict : dict
        Dictionary containing various features.

    """

    def __init__(
        self, wav_path: Path, name: str, number: int, sample_rate: int
    ) -> None:
        """Initialize a Presentation instance.

        Parameters
        ----------
        wav_path : Path
            Path to the WAV file.
        name : str
            Name of the presentation.
        number : int
            Number identifier for the presentation.
        sample_rate : int
            Sample rate of the audio file.

        """
        self.logger: Logger | None = None

        self.wav_path: Path = wav_path
        self.name: str = name
        self.number: int = number
        self.sample_rate: int = sample_rate

        self.is_preprocessed: bool = False

        self.active_intervals: list[tuple[float, float]] | None = None
        self.formants_df: pd.DataFrame | None = None
        self.whisper_result: WhisperResult | None = None
        self.yomi: str | None = None

        self.pyin_result: dict | None = None

    def set_logger(self, logger: Logger) -> None:
        """Set the logger for the presentation.

        Parameters
        ----------
        logger : Logger
            Logger for logging information.

        """
        self.logger = logger

    def calc_silence_intervals(
        self, silero_model: ScriptModule, silero_utils: list
    ) -> None:
        """Calculate the silence intervals in the audio by Silero VAD model."""
        # check if active_intervals is already calculated
        if self.active_intervals is not None:
            return

        get_speech_timestamps = silero_utils[0]
        read_audio = silero_utils[2]

        wav = read_audio(self.wav_path, sampling_rate=self.sample_rate)
        speech_dicts = get_speech_timestamps(
            wav, silero_model, sampling_rate=self.sample_rate
        )
        active_intervals = [
            (d["start"] / self.sample_rate, d["end"] / self.sample_rate)
            for d in speech_dicts
        ]
        self.active_intervals = active_intervals
        return

    def calc_formants(self) -> None:
        """Calculate formants using Praat and set a DataFrame."""
        # check if formants_df is already calculated
        if self.formants_df is not None:
            return

        if self.logger:
            self.logger.info("Start calculating formants")

        if self.active_intervals is None:
            error_message = (
                "self.active_intervals is None. "
                "Please run calc_silero_intervals first."
            )
            raise ValueError(error_message)
        path = self.wav_path

        sound = parselmouth.Sound(str(path))
        formant = sound.to_formant_burg()
        xs = formant.xs()

        formant_dicts = []
        interval_id = 0
        # record formants in each active interval
        for start, end in self.active_intervals:
            idx = 0
            for i in range(idx, len(xs)):
                if xs[i] < start:
                    continue
                if xs[i] > end:
                    idx = i
                    break
                d = {}
                d["t"] = xs[i]
                d["f1"] = formant.get_value_at_time(1, xs[i])
                d["f2"] = formant.get_value_at_time(2, xs[i])
                d["interval_id"] = interval_id
                formant_dicts.append(d)
            interval_id += 1
        # formants in active intervals
        formants_df = pd.DataFrame(formant_dicts)
        formants_df = formants_df.set_index("t")
        self.formants_df = formants_df
        return

    def whisper_transcribe(self, whisper_model: Whisper) -> None:
        """Transcribe the audio using the Whisper model."""
        # check if whisper_result is already calculated
        if self.whisper_result is not None:
            return
        whisper_result = whisper_model.transcribe(str(self.wav_path))
        whisper_result = whisper_model.align(
            str(self.wav_path), whisper_result, language="ja"
        )
        self.whisper_result = whisper_result
        self.yomi = get_yomi(whisper_result["text"])
        return

    def calc_pitch(self) -> None:
        """Calculate pitch using librosa's PYIN pitch extraction."""
        # check if pyin_result is already calculated
        if self.pyin_result is not None:
            return
        y, sampling_rate = librosa.load(self.wav_path, sr=self.sample_rate)
        # fmin, fmaxはLuzardoの論文と同じ.
        # Luzardo論文では20ms window size
        # 今回は1frame = 1000/16000 ms なので, 20msのためには20/1000 * 16000 = 320frame
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=60,
            fmax=400,
            sr=int(sampling_rate),
            frame_length=int(20 / 1000 * sampling_rate),
        )
        d = {}
        d["f0"] = f0
        d["voiced_flag"] = voiced_flag
        d["voiced_probs"] = voiced_probs
        self.pyin_result = d
        return

    def preprocess(self, model_container: ModelContainer) -> None:
        """Preprocess the audio using the given models."""
        silero_model = model_container.silero_model
        silero_utils = model_container.silero_utils
        whisper_model = model_container.whisper_model
        if silero_model is None or silero_utils is None or whisper_model is None:
            error_message = "Models must be loaded before preprocessing."
            raise ValueError(error_message)

        self.calc_silence_intervals(silero_model, silero_utils)
        self.calc_formants()
        self.whisper_transcribe(whisper_model)
        self.calc_pitch()
        self.is_preprocessed = True


class FeatureHandler:
    """Calculate various features from a preprocessed Presentation object."""

    @staticmethod
    def calc_formant_rates(
        p: Presentation, num_frame: int = 19, thresh: int = 100
    ) -> dict:
        """Calculate formant stability rates.

        今回は1フレーム0.00625秒 = 6.25ms
        Luzardo et al. では20ms x 6frameでフォルマントの標準偏差を計算
        120 / 6.25 = 19.2 frameより, 19frameがデフォルト値
        threshもLuzardo et al. と同じ100をデフォルト値としている

        Args:
        ----
            p (Presentation): Presentation object.
            num_frame (int, optional): length of the rolling window.  Defaults to 19.
            thresh (int, optional): threshold for stability rate. Defaults to 100.

        Returns:
        -------
            dict: dictionary containing formant stability rates.

        """
        formants_df = p.formants_df
        if formants_df is None:
            msg = "formants_df is None. Please run calc_formants_df first."
            raise ValueError(msg)
        gs = []
        for _, g in formants_df.groupby("interval_id"):
            for i in [1, 2]:
                g[f"rolling_std_f{i}"] = g[f"f{i}"].rolling(num_frame).std()
            gs.append(g)
        formants_df = pd.concat(gs)
        rolling_std_f1 = formants_df["rolling_std_f1"].dropna().to_numpy()
        rolling_std_f2 = formants_df["rolling_std_f2"].dropna().to_numpy()
        f1r = len(rolling_std_f1[rolling_std_f1 < thresh]) / len(rolling_std_f1)
        f2r = len(rolling_std_f2[rolling_std_f2 < thresh]) / len(rolling_std_f2)
        f1f2r = (f1r + f2r) / 2

        return {
            f"F1R_{thresh}": f1r,
            f"F2R_{thresh}": f2r,
            f"F1F2R_{thresh}": f1f2r,
        }

    @staticmethod
    def calc_pitch_features(p: Presentation) -> dict:
        """Calculate pitch related features."""
        if p.pyin_result is None:
            msg = "pyin_result is None. Please run calc_pitch first."
            raise ValueError(msg)
        f0 = p.pyin_result["f0"]
        voiced_flag = p.pyin_result["voiced_flag"]
        f0 = f0[voiced_flag]
        avgp = np.log(f0).mean()
        stdp = np.log(f0).std()
        # 0.1 and 0.9 quantiles
        q1 = np.quantile(f0, 0.1)
        q9 = np.quantile(f0, 0.9)

        return {
            "AVGP": avgp,
            "STDP": stdp,
            "Q1P": q1,
            "Q9P": q9,
        }

    @staticmethod
    def calc_articulation_rate(p: Presentation) -> dict:
        """Calculate articulation rate."""
        formants_df = p.formants_df
        if formants_df is None:
            msg = "formants_df is None. Please run calc_formants_df first."
            raise ValueError(msg)
        tmp_df = (
            formants_df.reset_index().groupby("interval_id")["t"].agg(["min", "max"])
        )
        durs = (tmp_df["min"].shift(-1) - tmp_df["max"]).to_numpy()[:-1]
        total_time = formants_df.index[-1] - formants_df.index[0]
        total_speech_time = total_time - durs.sum()
        if p.yomi is None:
            msg = "yomi is None. Please run whisper_transcribe first."
            raise ValueError(msg)
        articulation_rate = len(p.yomi) / total_speech_time

        return {"AR": articulation_rate}

    @staticmethod
    def calc_volume_variation(p: Presentation) -> dict:
        """Calculate volume variation."""
        path = p.wav_path
        y, sr = librosa.load(path)
        rms = librosa.feature.rms(y=y)
        ts = librosa.times_like(rms, sr=sr)

        vol_dicts = []
        interval_id = 0
        # record formants in each active interval
        if p.active_intervals is None:
            error_message = (
                "p.active_intervals is None. " "Please run calc_silero_intervals first."
            )
            raise ValueError(error_message)
        for start, end in p.active_intervals:
            idx = 0
            for i in range(idx, len(ts)):
                if ts[i] < start:
                    continue
                if ts[i] > end:
                    idx = i
                    break
                d = {}
                d["t"] = ts[i]
                d["vol"] = rms[0, i]
                d["interval_id"] = interval_id
                vol_dicts.append(d)
            interval_id += 1
        # formants in active intervals
        vol_df = pd.DataFrame(vol_dicts)
        vol_df = vol_df.set_index("t")

        # volume variation in each active interval
        vol_stds = []
        vol_means = []
        for _, g in vol_df.groupby("interval_id"):
            vol_stds.append(g["vol"].std())
            vol_means.append(g["vol"].mean())
        vol_var_local = np.mean(vol_stds)
        vol_var_global = np.std(vol_means)

        return {
            "VOL_VAR_LOCAL": vol_var_local,
            "VOL_VAR_GLOBAL": vol_var_global,
        }

    @staticmethod
    def calc_silence(p: Presentation) -> dict:
        """Calculate silence related features."""
        if p.formants_df is None:
            msg = "formants_df is None. Please run calc_formants_df first."
            raise ValueError(msg)
        tmp_df = (
            p.formants_df.reset_index().groupby("interval_id")["t"].agg(["min", "max"])
        )
        durs = (tmp_df["min"].shift(-1) - tmp_df["max"]).to_numpy()[:-1]
        silence_time_ratio = durs.sum() / (
            p.formants_df.index[-1] - p.formants_df.index[0]
        )
        th = 0.5
        short_silence_rate = len(durs[durs < th]) / len(durs)

        return {
            "SILENCE_TIME_RATIO": silence_time_ratio,
            "SHORT_SILENCE_RATE": short_silence_rate,
        }

    @staticmethod
    def calc_detailed_speeds(p: Presentation) -> dict:
        """Calculate detailed speed related features."""
        if p.whisper_result is None:
            msg = "whisper_result is None. Please run whisper_transcribe first."
            raise ValueError(msg)
        whisper_d = p.whisper_result.to_dict()
        speeds = []
        starts, ends = [], []
        for segment in whisper_d["segments"]:
            s, e = segment["start"], segment["end"]
            yomi = get_yomi(segment["text"])
            speed = len(yomi) / (e - s)
            speeds.append(speed)
            starts.append(s)
            ends.append(e)
        speeds = np.array(speeds)
        starts = np.array(starts)
        ends = np.array(ends)

        th = 20  # 20文字/sより速い区間は異常値として除外
        speeds = speeds[speeds < th]
        avg_speed = speeds.mean()
        std_speed = speeds.std()

        silence_duration = starts[1:] - ends[:-1]

        return {
            "AVG_SPEED": avg_speed,
            "STD_SPEED": std_speed,
            "AVG_SILENCE_DURATION_SENTENCE": silence_duration.mean(),
            "STD_SILENCE_DURATION_SENTENCE": silence_duration.std(),
        }
