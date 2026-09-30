"""Compare facial emotion recognition with turn duration.

Transcript timestamps are in seconds; FER timestamps use HH:MM:SS. FER
observations are assigned to transcript turns in the interval [start, stop).
Both hypotheses use the same rows and log turn duration as the outcome.

H1 uses emotion probabilities, omitting neutral as the reference component.
H2 uses the dominant emotion, with neutral as the reference category. Models
include dialog and speaker random effects. If no optimizer converges, the
analysis stops rather than falling back to OLS.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config.local import OUTPUT_DIR, RAW_MEDIA_DIR

# Configuration
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Exact filename of the FER file.
FER_FILENAME = "audio_video_features.csv"

# Exact filename of the transcript.
TRANSCRIPT_FILENAME = "transcript_cliffhanger.csv"

# Transcript timestamps are seconds.
TRANSCRIPT_TIME_UNIT = "s"

# FER timestamps are stored in the timedelta column.
FER_TIME_COLUMN = "timedelta"

# FER participant identifier.
FER_SPEAKER_COLUMN = "user_id"

# At least this many FER observations must fall inside a turn.
MIN_FER_OBSERVATIONS = 1

# Optimizers that are tried for the MixedLM.
#
# The script stops if none of these produces a converged model.
MIXED_MODEL_OPTIMIZERS = ["lbfgs", "bfgs", "cg", "powell"]

# Maximum number of iterations for optimizers.
MIXED_MODEL_MAXITER = 2000

# Emotions
EMOTIONS = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

FER_EMOTION_COLUMNS = {
    "anger": "prob_face_anger",
    "contempt": "prob_face_contempt",
    "disgust": "prob_face_disgust",
    "fear": "prob_face_fear",
    "happiness": "prob_face_happiness",
    "neutral": "prob_face_neutral",
    "sadness": "prob_face_sadness",
    "surprise": "prob_face_surprise",
}

# Backchannel phrases
PURE_BACKCHANNELS = {
    "uh huh",
    "uh-huh",
    "uhuh",
    "mhm",
    "mmhm",
    "mm hmm",
    "mm-hmm",
    "m hm",
    "hmm",
    "hm",
    "mm",
    "mmm",
    "yeah",
    "yes",
    "yep",
    "yup",
    "right",
    "okay",
    "ok",
    "oh",
    "sure",
}

PURE_BACKCHANNEL_PHRASES = {
    "yeah yeah",
    "yes yes",
    "okay okay",
    "oh yeah",
    "right right",
    "uh huh",
    "mm hmm",
    "mhm mhm",
    "yeah mhm",
    "right mhm",
    "okay mhm",
    "yeah okay",
    "okay yeah",
}


# Shared helpers


def clean_utterance(value) -> str:
    """Normalize transcript utterance text."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def to_bool(series: pd.Series) -> pd.Series:
    """
    Convert different boolean representations to True/False.

    Handles:
        True / False
        1 / 0
        yes / no
        WAHR / FALSCH
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"true", "1", "yes", "y", "wahr"})


def normalize_backchannel_text(text: str) -> str:
    """Normalize text before backchannel classification."""
    text = clean_utterance(text).lower()
    text = re.sub(r"[^\w\s'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_pure_backchannel(text: str) -> bool:
    """
    Return True only when the complete utterance is a backchannel.

    This intentionally does NOT classify every one-word turn as a
    backchannel.

    Therefore:

        "Why?"
        "How?"
        "Work."
        "Really?"

    are retained.
    """
    normalized = normalize_backchannel_text(text)
    if not normalized:
        return False
    if normalized in PURE_BACKCHANNELS:
        return True
    if normalized in PURE_BACKCHANNEL_PHRASES:
        return True
    tokens = normalized.split()
    # Examples:
    #
    #     yeah yeah
    #     yeah yeah yeah
    #
    return 2 <= len(tokens) <= 6 and all(token in PURE_BACKCHANNELS for token in tokens)


# Paths


def find_transcript_files() -> list[Path]:
    """
    Find every transcript_cliffhanger.csv below RAW_MEDIA_DIR.
    """
    if not RAW_MEDIA_DIR.exists():
        raise FileNotFoundError(f"RAW_MEDIA_DIR does not exist:\n{RAW_MEDIA_DIR}")
    files = sorted(path for path in RAW_MEDIA_DIR.rglob(TRANSCRIPT_FILENAME) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No '{TRANSCRIPT_FILENAME}' files found below:\n{RAW_MEDIA_DIR}")
    print()
    print("=" * 80)
    print("TRANSCRIPT FILES")
    print("=" * 80)
    print(f"Found {len(files):,} transcript files.")
    return files


def get_dialog_dir(transcript_path: Path) -> Path:
    """
    Get the dialog directory.

    Expected structure:

        dialog/
            transcript/
                transcript_cliffhanger.csv

    Therefore:

        transcript_path.parent
            -> transcript

        transcript_path.parent.parent
            -> dialog
    """
    if transcript_path.parent.name.lower() in {"transcript", "transcription"}:
        return transcript_path.parent.parent
    return transcript_path.parent


def get_dialog_id(transcript_path: Path) -> str:
    """The dialog directory name is used as dialog_id."""
    return get_dialog_dir(transcript_path).name


def find_fer_file(dialog_dir: Path) -> Path:
    """
    Find audio_video_features.csv only inside the current dialog directory.
    """
    fer_path = dialog_dir / FER_FILENAME
    if not fer_path.exists():
        raise FileNotFoundError(f"FER file not found:\n{fer_path}")
    return fer_path


# Transcript loading


def load_transcript(transcript_path: Path) -> pd.DataFrame:
    """Load one transcript_cliffhanger.csv."""
    df = pd.read_csv(transcript_path)
    required_columns = [
        "turn_id",
        "speaker",
        "start",
        "stop",
        "utterance",
        "questions",
        "end_question",
        "overlap",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing transcript columns in {transcript_path}:\n"
            + f"{missing}\n\nAvailable columns:\n{list(df.columns)}"
        )
    df = df.copy()
    # Basic cleaning
    df["utterance"] = df["utterance"].map(clean_utterance)
    df["turn_id"] = pd.to_numeric(df["turn_id"], errors="coerce")
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["stop"] = pd.to_numeric(df["stop"], errors="coerce")
    df = df.dropna(subset=["turn_id", "start", "stop", "speaker"]).copy()
    # Time
    if TRANSCRIPT_TIME_UNIT == "ms":
        time_scale = 1000.0
    elif TRANSCRIPT_TIME_UNIT == "s":
        time_scale = 1.0
    else:
        raise ValueError("Unsupported transcript time unit: " + f"{TRANSCRIPT_TIME_UNIT!r}. Use 's' or 'ms'.")
    df["start_sec"] = df["start"] / time_scale
    df["stop_sec"] = df["stop"] / time_scale
    df["turn_duration"] = df["stop_sec"] - df["start_sec"]
    # Boolean variables
    df["questions"] = to_bool(df["questions"])
    df["end_question"] = to_bool(df["end_question"])
    df["overlap"] = to_bool(df["overlap"])
    # Dialog information
    dialog_dir = get_dialog_dir(transcript_path)
    df["dialog_id"] = dialog_dir.name
    df["source_transcript"] = str(transcript_path)
    # Sort.
    #
    # conversation_position is deliberately NOT calculated here.
    #
    # It is calculated later, after:
    #
    #     1. pre-dialog filtering
    #     2. backchannel filtering
    #
    # This ensures that position refers to the actual final analysis turns.
    df = df.sort_values(["start_sec", "stop_sec", "turn_id"]).reset_index(drop=True)
    return df


def load_all_transcripts() -> pd.DataFrame:
    """Load every transcript in RAW_MEDIA_DIR."""
    transcript_files = find_transcript_files()
    frames = []
    for path in transcript_files:
        try:
            df = load_transcript(path)
            frames.append(df)
            print(f"OK: {get_dialog_id(path)} -> {len(df):,} turns")
        except Exception as exc:
            print("WARNING: Could not load transcript:")
            print(f"  {path}")
            print(f"  Error: {exc}")
    if not frames:
        raise RuntimeError("No transcript files could be loaded.")
    result = pd.concat(frames, ignore_index=True)
    print()
    print(f"Dialogs: {result['dialog_id'].nunique():,}")
    print(f"Turns:   {len(result):,}")
    return result


# Pre-dialog filtering


def remove_pre_dialog_turns(turns: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the initial section where only one participant is speaking.

    The first complete turn after the second participant appears is retained.

    This is performed before conversation_position is calculated.
    """
    print()
    print("=" * 80)
    print("REMOVING PRE-DIALOG MATERIAL")
    print("=" * 80)
    result = []
    for dialog_id, group in turns.groupby("dialog_id", sort=False):
        group = group.sort_values(["start_sec", "stop_sec", "turn_id"]).copy()
        speakers_seen = set()
        cutoff = None
        for _, row in group.iterrows():
            speaker = str(row["speaker"])
            speakers_seen.add(speaker)
            if len(speakers_seen) >= 2:
                cutoff = row["start_sec"]
                break
        if cutoff is None:
            print(f"WARNING: {dialog_id}: only one speaker found.")
            result.append(group)
            continue
        before = len(group)
        group = group[group["start_sec"] >= cutoff].copy()
        removed = before - len(group)
        print(f"{dialog_id}: removed {removed} pre-dialog turn(s)")
        result.append(group)
    if not result:
        raise RuntimeError("No turns remain after pre-dialog filtering.")
    return pd.concat(result, ignore_index=True)


# Backchannel filtering


def apply_backchannel_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove genuine pure backchannels.

    This is deliberately performed BEFORE FER matching.
    """
    print()
    print("=" * 80)
    print("BACKCHANNEL FILTER")
    print("=" * 80)
    df = df.copy()
    df["pure_backchannel"] = df["utterance"].map(is_pure_backchannel)
    before = len(df)
    removed = df[df["pure_backchannel"]].copy()
    print(f"Turns before: {before:,}")
    print(f"Pure backchannels removed: {len(removed):,}")
    if not removed.empty:
        print()
        print("Examples:")
        for text in removed["utterance"].head(15):
            print(f"  - {text!r}")
    result = df[~df["pure_backchannel"]].copy()
    print(f"Turns after: {len(result):,}")
    return result.reset_index(drop=True)


# Conversation position


def recompute_conversation_position(turns: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute conversation position after all turn-level filtering.

    Position is normalized within each dialog:

        first retained turn  = 0
        last retained turn   = 1

    If a dialog contains only one retained turn,
    its position is set to 0.
    """
    print()
    print("=" * 80)
    print("RECOMPUTING CONVERSATION POSITION")
    print("=" * 80)
    result = []
    for dialog_id, group in turns.groupby("dialog_id", sort=False):
        group = group.sort_values(["start_sec", "stop_sec", "turn_id"]).copy()
        n = len(group)
        if n <= 1:
            group["conversation_position"] = 0.0
        else:
            group["conversation_position"] = np.arange(n) / (n - 1)
        result.append(group)
    if not result:
        raise RuntimeError("No turns available for conversation_position.")
    result = pd.concat(result, ignore_index=True)
    print(f"Dialogs: {result['dialog_id'].nunique():,}")
    print(f"Turns:   {len(result):,}")
    position_min = result["conversation_position"].min()
    position_max = result["conversation_position"].max()
    print(f"Position range: {position_min:.3f} - " + f"{position_max:.3f}")
    return result


# FER timestamps


def timedelta_to_seconds(series: pd.Series) -> pd.Series:
    """
    Convert FER timedelta values to seconds.

    Examples:

        00:00:00
        00:00:01
        00:00:02
        00:01:37
        01:02:03
    """
    values = pd.to_timedelta(series.astype(str), errors="coerce")
    return values.dt.total_seconds()


# FER loading


def load_fer_for_dialog(dialog_dir: Path) -> pd.DataFrame:
    """
    Load audio_video_features.csv for exactly one dialog.
    """
    fer_path = find_fer_file(dialog_dir)
    print(f"  FER: {fer_path}")
    fer = pd.read_csv(fer_path, low_memory=False)
    required_columns = [FER_TIME_COLUMN, FER_SPEAKER_COLUMN] + list(FER_EMOTION_COLUMNS.values())
    missing = [column for column in required_columns if column not in fer.columns]
    if missing:
        raise ValueError(
            f"Missing FER columns in:\n{fer_path}\n\nMissing:\n{missing}\n\n" + f"Available:\n{list(fer.columns)}"
        )
    fer = fer.copy()
    # Time
    fer["fer_time_sec"] = timedelta_to_seconds(fer[FER_TIME_COLUMN])
    # Speaker
    fer["fer_speaker"] = fer[FER_SPEAKER_COLUMN].astype(str).str.strip()
    # Emotion probabilities
    for emotion, source_column in FER_EMOTION_COLUMNS.items():
        fer[emotion] = pd.to_numeric(fer[source_column], errors="coerce")
    # Keep only relevant columns.
    keep_columns = ["fer_time_sec", "fer_speaker"] + EMOTIONS
    fer = fer[keep_columns].copy()
    # Remove observations without timestamps.
    fer = fer.dropna(subset=["fer_time_sec"])
    # Remove observations where all emotion probabilities are missing.
    fer = fer.dropna(subset=EMOTIONS, how="all")
    # Diagnostics for emotion completeness.
    complete_probability_rows = fer[EMOTIONS].notna().all(axis=1).sum()
    incomplete_probability_rows = len(fer) - complete_probability_rows
    print(f"  FER rows: {len(fer):,}")
    print(f"  FER speakers: {fer['fer_speaker'].nunique():,}")
    print(f"  Complete emotion rows: {complete_probability_rows:,}")
    print(f"  Incomplete emotion rows: {incomplete_probability_rows:,}")
    if fer["fer_time_sec"].notna().any():
        time_min = fer["fer_time_sec"].min()
        time_max = fer["fer_time_sec"].max()
        print(f"  FER time range: {time_min:.3f}s - " + f"{time_max:.3f}s")
    return fer.sort_values(["fer_speaker", "fer_time_sec"]).reset_index(drop=True)


def load_all_fer(turns: pd.DataFrame) -> pd.DataFrame:
    """
    Load one audio_video_features.csv per dialog.
    """
    print()
    print("=" * 80)
    print("LOADING FER")
    print("=" * 80)
    frames = []
    dialogs = turns[["dialog_id", "source_transcript"]].drop_duplicates("dialog_id")
    for _, row in dialogs.iterrows():
        dialog_id = row["dialog_id"]
        transcript_path = Path(row["source_transcript"])
        dialog_dir = get_dialog_dir(transcript_path)
        print()
        print(f"Dialog: {dialog_id}")
        try:
            fer = load_fer_for_dialog(dialog_dir)
            fer["dialog_id"] = dialog_id
            frames.append(fer)
        except Exception as exc:
            print(f"WARNING: FER could not be loaded for {dialog_id}")
            print(f"  {exc}")
    if not frames:
        raise RuntimeError("No FER files could be loaded.")
    result = pd.concat(frames, ignore_index=True)
    print()
    print(f"Total FER observations: {len(result):,}")
    print(f"Dialogs with FER: {result['dialog_id'].nunique():,}")
    return result


# Matching FER to turns


def aggregate_fer_to_turns(turns: pd.DataFrame, fer: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate FER observations inside each complete transcript turn.

    Matching conditions:

        same dialog
        AND
        same speaker
        AND
        turn start <= FER time < turn stop

    IMPORTANT:

        The interval is [start, stop), not [start, stop].

        This prevents an FER observation exactly at a turn boundary
        from being assigned to both adjacent turns.
    """
    print()
    print("=" * 80)
    print("MATCHING FER TO TURNS")
    print("=" * 80)
    records = []
    # Pre-group FER for efficient lookup.
    fer_groups = {
        (str(dialog_id), str(speaker)): group.sort_values("fer_time_sec")
        for (dialog_id, speaker), group in fer.groupby(["dialog_id", "fer_speaker"], sort=False)
    }
    # Match every transcript turn.
    for _, turn in turns.iterrows():
        dialog_id = str(turn["dialog_id"])
        speaker = str(turn["speaker"])
        key = (dialog_id, speaker)
        group = fer_groups.get(key)
        # No FER data for this speaker.
        if group is None:
            records.append({**{emotion: np.nan for emotion in EMOTIONS}, "fer_n": 0})
            continue
        # FER observations inside the turn.
        #
        # [start, stop)
        mask = (group["fer_time_sec"] >= turn["start_sec"]) & (group["fer_time_sec"] < turn["stop_sec"])
        matched = group.loc[mask, EMOTIONS].copy()
        # Keep only complete FER observations.
        #
        # This is important because the emotion probabilities form a
        # compositional vector. We do not want the mean for one emotion
        # to be based on a different set of frames than another emotion.
        matched = matched.dropna(how="any")
        # Not enough complete FER observations.
        if len(matched) < MIN_FER_OBSERVATIONS:
            records.append(
                {
                    **{emotion: np.nan for emotion in EMOTIONS},
                    "fer_n": len(matched),
                }
            )
            continue
        # Mean probability over the complete turn.
        means = matched.mean(numeric_only=True)
        record = {emotion: means[emotion] for emotion in EMOTIONS}
        record["fer_n"] = len(matched)
        records.append(record)
    # Combine transcript and FER.
    fer_turns = pd.DataFrame(records, index=turns.index)
    result = pd.concat([turns, fer_turns], axis=1)
    # Normalize emotion probabilities.
    #
    # Because we now use complete FER rows, the mean probability vector
    # should already sum to approximately 1.
    #
    # We nevertheless normalize to protect against tiny numerical deviations.
    complete_emotions = result[EMOTIONS].notna().all(axis=1)
    probability_sum = result.loc[complete_emotions, EMOTIONS].sum(axis=1)
    valid = complete_emotions & probability_sum.gt(0)
    result.loc[valid, EMOTIONS] = result.loc[valid, EMOTIONS].div(probability_sum[valid], axis=0)
    # Diagnostics.
    n_with_fer = (result["fer_n"] >= MIN_FER_OBSERVATIONS).sum()
    n_without_fer = (result["fer_n"] < MIN_FER_OBSERVATIONS).sum()
    print()
    print(f"Turns with FER:    {n_with_fer:,}")
    print(f"Turns without FER: {n_without_fer:,}")
    print(f"FER coverage: {n_with_fer / len(result) * 100:.2f}%")
    print()
    print("FER observations per turn:")
    print(result["fer_n"].describe())
    return result.reset_index(drop=True)


# Model dataset


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the final dataset used identically by H1 and H2.

    Both hypotheses therefore operate on exactly the same rows.
    """
    print()
    print("=" * 80)
    print("PREPARING MODEL DATA")
    print("=" * 80)
    result = df.copy()
    before = len(result)
    # Complete FER probability vector.
    result = result.dropna(subset=EMOTIONS).copy()
    # Positive turn duration.
    result = result[result["turn_duration"] > 0].copy()
    # Valid conversation position.
    result = result.dropna(subset=["conversation_position"]).copy()
    # Log turn duration.
    result["log_turn_duration"] = np.log(result["turn_duration"])
    # Dominant emotion.
    result["dominant_emotion"] = result[EMOTIONS].idxmax(axis=1)
    result["dominant_emotion"] = pd.Categorical(result["dominant_emotion"], categories=EMOTIONS)
    # Boolean controls -> integers.
    result["questions"] = result["questions"].astype(int)
    result["end_question"] = result["end_question"].astype(int)
    result["overlap"] = result["overlap"].astype(int)
    # Final sanity checks.
    if result.empty:
        raise RuntimeError("No rows remain after model-data preparation.")
    if not np.isfinite(result["log_turn_duration"]).all():
        raise RuntimeError("Non-finite values detected in log_turn_duration.")
    print(f"Rows before model cleaning: {before:,}")
    print(f"Rows used in models: {len(result):,}")
    print(f"Rows removed: {before - len(result):,}")
    print()
    print(f"Dialogs: {result['dialog_id'].nunique():,}")
    print(f"Speakers: {result['speaker'].nunique():,}")
    print()
    print("Dominant emotions:")
    print(result["dominant_emotion"].value_counts())
    print()
    print("Turn duration:")
    print(result["turn_duration"].describe())
    print()
    print("FER observations per turn:")
    print(result["fer_n"].describe())
    return result.reset_index(drop=True)


# Mixed-effects models


def fit_mixed_model(formula: str, data: pd.DataFrame, model_name: str):
    """
    Fit a MixedLM with:

        random intercept:
            dialog

        additional variance component:
            speaker

    Several optimizers are attempted.

    IMPORTANT:

        There is NO automatic OLS fallback.

        If no optimizer converges, the function raises RuntimeError.
    """
    print()
    print("=" * 80)
    print(f"FITTING MIXED MODEL: {model_name}")
    print("=" * 80)
    # Basic grouping diagnostics.
    n_rows = len(data)
    n_dialogs = data["dialog_id"].nunique()
    n_speakers = data["speaker"].nunique()
    dialog_counts = data["dialog_id"].value_counts()
    speaker_counts = data["speaker"].value_counts()
    repeated_dialogs = (dialog_counts > 1).sum()
    repeated_speakers = (speaker_counts > 1).sum()
    print()
    print("Model data:")
    print(f"  Rows:                {n_rows:,}")
    print(f"  Dialogs:             {n_dialogs:,}")
    print(f"  Speakers:            {n_speakers:,}")
    print(f"  Repeated dialogs:    {repeated_dialogs:,}")
    print(f"  Repeated speakers:   {repeated_speakers:,}")
    # Construct MixedLM.
    #
    # dialog is the main grouping variable.
    #
    # speaker is represented as an additional variance component.
    model = smf.mixedlm(
        formula=formula,
        data=data,
        groups=data["dialog_id"],
        vc_formula={"speaker": "0 + C(speaker)"},
        re_formula="1",
    )
    optimizer_results = []
    # Try optimizers.
    for optimizer in MIXED_MODEL_OPTIMIZERS:
        print()
        print(f"Trying optimizer: {optimizer}")
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")

                result = model.fit(
                    reml=True,
                    method=optimizer,
                    maxiter=MIXED_MODEL_MAXITER,
                    disp=False,
                )
            converged = bool(getattr(result, "converged", False))
            print(f"  converged = {converged}")
            if caught_warnings:
                convergence_warnings = [
                    str(w.message)
                    for w in caught_warnings
                    if (
                        "converg" in str(w.message).lower()
                        or "singular" in str(w.message).lower()
                        or "boundary" in str(w.message).lower()
                    )
                ]
                for warning_message in convergence_warnings:
                    print(f"  warning: {warning_message}")
            optimizer_results.append(
                {
                    "optimizer": optimizer,
                    "converged": converged,
                    "result": result,
                }
            )
            if converged:
                print()
                print(f"SUCCESS: {model_name} converged with {optimizer}.")
                result._analysis_optimizer = optimizer
                return result
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            optimizer_results.append(
                {
                    "optimizer": optimizer,
                    "converged": False,
                    "result": None,
                    "error": repr(exc),
                }
            )
    # Nothing converged.
    #
    # DO NOT silently switch to OLS.
    print()
    print("=" * 80)
    print("MIXED MODEL DID NOT CONVERGE")
    print("=" * 80)
    print()
    print(f"Model: {model_name}")
    print()
    print("Optimizer results:")
    for attempt in optimizer_results:
        optimizer = attempt["optimizer"]
        converged = attempt["converged"]
        print(f"  {optimizer}: converged={converged}")
        if not converged and attempt.get("error"):
            print(f"      error: {attempt['error']}")
    raise RuntimeError(
        f"{model_name} could not be fitted with a converged MixedLM. " + "No OLS fallback was performed."
    )


# Model output


def save_model_outputs(result, prefix: str):
    """Save model summary and coefficient table."""
    summary_path = OUTPUT_DIR / f"{prefix}_summary.txt"
    coef_path = OUTPUT_DIR / f"{prefix}_coefficients.csv"
    # Summary.
    summary_text = str(result.summary())
    optimizer = getattr(result, "_analysis_optimizer", "unknown")
    summary_text = (
        f"Analysis optimizer: {optimizer}\n" + f"Converged: {getattr(result, 'converged', False)}\n\n" + summary_text
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    # Coefficients.
    confidence_intervals = result.conf_int()
    coefficients = pd.DataFrame(
        {
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "z": result.tvalues.values,
            "p_value": result.pvalues.values,
            "ci_low": confidence_intervals[0].values,
            "ci_high": confidence_intervals[1].values,
        }
    )
    coefficients.to_csv(coef_path, index=False)
    print()
    print(f"Saved: {summary_path}")
    print(f"Saved: {coef_path}")


# H1


def run_h1(data: pd.DataFrame):
    """
    H1:

    Continuous facial emotion probabilities.

    Neutral is omitted because the eight probabilities sum to 1.

    Interpretation:

        The included coefficients represent associations of the
        respective non-neutral emotion probabilities relative to the
        implicit neutral component.

    The dependent variable is:

        log_turn_duration
    """
    print()
    print("=" * 80)
    print("H1 / CONTINUOUS FER PROBABILITIES")
    print("=" * 80)
    model_data = data.copy()
    # Create explicit formula variables.
    for emotion in EMOTIONS:
        model_data[f"prob_face_{emotion}"] = model_data[emotion]
    # Neutral is intentionally omitted.
    formula = """
        log_turn_duration
        ~ prob_face_anger
        + prob_face_contempt
        + prob_face_disgust
        + prob_face_fear
        + prob_face_happiness
        + prob_face_sadness
        + prob_face_surprise
        + conversation_position
        + questions
        + end_question
        + overlap
    """

    formula = re.sub(r"\s+", " ", formula).strip()
    result = fit_mixed_model(formula=formula, data=model_data, model_name="H1")
    print()
    print(result.summary())
    save_model_outputs(result, "H1_model1_continuous")
    return result


# H2


def run_h2(data: pd.DataFrame):
    """
    H2:

    Dominant facial emotion.

    Neutral is the explicit reference category.

    The dependent variable is:

        log_turn_duration
    """
    print()
    print("=" * 80)
    print("H2 / DOMINANT FACIAL EMOTION")
    print("=" * 80)
    model_data = data.copy()
    # Remove unused categories.
    model_data["dominant_emotion"] = model_data["dominant_emotion"].cat.remove_unused_categories()
    # Explicit neutral reference.
    formula = """
        log_turn_duration
        ~ C(
            dominant_emotion,
            Treatment(reference='neutral')
        )
        + conversation_position
        + questions
        + end_question
        + overlap
    """

    formula = re.sub(r"\s+", " ", formula).strip()
    result = fit_mixed_model(formula=formula, data=model_data, model_name="H2")
    print()
    print(result.summary())
    save_model_outputs(result, "H2_model2_dominant")
    return result


# Dataset diagnostics


def run_dataset_diagnostics(data: pd.DataFrame):
    """
    Print diagnostics for the final common model dataset.

    H1 and H2 must receive this exact same dataset.
    """
    print()
    print("=" * 80)
    print("FINAL DATASET DIAGNOSTICS")
    print("=" * 80)
    print()
    print(f"Rows: {len(data):,}")
    print(f"Dialogs: {data['dialog_id'].nunique():,}")
    print(f"Speakers: {data['speaker'].nunique():,}")
    print()
    print("Rows per dialog:")
    print(data["dialog_id"].value_counts().describe())
    print()
    print("Rows per speaker:")
    print(data["speaker"].value_counts().describe())
    print()
    print("FER observations per turn:")
    print(data["fer_n"].describe())
    print()
    print("Turn duration:")
    print(data["turn_duration"].describe())
    print()
    print("Log turn duration:")
    print(data["log_turn_duration"].describe())
    print()
    print("Emotion probability sums:")
    probability_sums = data[EMOTIONS].sum(axis=1)
    print(probability_sums.describe())
    max_deviation = np.max(np.abs(probability_sums - 1.0))
    print()
    print(f"Maximum deviation from probability sum = 1: {max_deviation:.12f}")


# Main workflow


def main():
    print("=" * 80)
    print("H1 / H2 TURN-TAKING + FER ANALYSIS")
    print("=" * 80)
    print()
    print(f"Raw media directory:\n{RAW_MEDIA_DIR}")
    print()
    print(f"Output directory:\n{OUTPUT_DIR}")
    # Load every transcript.
    turns = load_all_transcripts()
    # Remove initial pre-dialog material.
    turns = remove_pre_dialog_turns(turns)
    # Remove pure backchannels.
    # This happens BEFORE FER matching.
    turns = apply_backchannel_filter(turns)
    # Recompute conversation position.
    # This happens AFTER all turn filtering.
    turns = recompute_conversation_position(turns)
    # Load corresponding FER file for every dialog.
    fer = load_all_fer(turns)
    # Match FER observations to complete transcript turns.
    # Matching interval:
    #     [start, stop)
    merged = aggregate_fer_to_turns(turns, fer)
    # Prepare ONE common model dataset.
    # H1 and H2 receive exactly the same rows.
    model_data = prepare_model_data(merged)
    # Final diagnostics.
    run_dataset_diagnostics(model_data)
    # Save turn-level dataset.
    merged_path = OUTPUT_DIR / "turn_level_fer_dataset.csv"
    merged.to_csv(merged_path, index=False)
    print()
    print(f"Saved:\n{merged_path}")
    # Save common model dataset.
    model_data_path = OUTPUT_DIR / "model_dataset.csv"
    model_data.to_csv(model_data_path, index=False)
    print()
    print(f"Saved:\n{model_data_path}")
    # H1
    h1_result = run_h1(model_data)
    # H2
    h2_result = run_h2(model_data)
    # Final overview.
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print()
    print(f"Final model rows: {len(model_data):,}")
    print(f"Dialogs: {model_data['dialog_id'].nunique():,}")
    print(f"Speakers: {model_data['speaker'].nunique():,}")

    print()
    print("H1: continuous FER probabilities")
    print("H2: dominant facial emotion")

    print()
    print("H1 reference structure:")
    print("  Neutral omitted because emotion probabilities sum to 1.")

    print()
    print("H2 reference category:")
    print("  Neutral")

    print()
    print("Random effects:")
    print("  Dialog random intercept")
    print("  Speaker variance component")

    print()
    print("H1 optimizer:")
    print(f"  {getattr(h1_result, '_analysis_optimizer', 'unknown')}")
    print("H1 converged:")
    print(f"  {getattr(h1_result, 'converged', False)}")

    print()
    print("H2 optimizer:")
    print(f"  {getattr(h2_result, '_analysis_optimizer', 'unknown')}")
    print("H2 converged:")
    print(f"  {getattr(h2_result, 'converged', False)}")

    print()
    print(f"Output directory:\n{OUTPUT_DIR}")


# Script entry point


if __name__ == "__main__":
    main()
