# candor-fer-turn-taking
Analysis of facial emotion and turn-taking using the CANDOR conversation corpus


# H1 / H2 Turn-Level Analysis
===========================

Analysis of the relationship between facial emotion recognition (FER)
and turn duration in conversational data.

 Dataset structure
-----------------

F:\\raw_media
├── raw_media_part_1
│   ├── <dialog_id>
│   │   ├── transcript
│   │   │   └── transcript_cliffhanger.csv
│   │   └── audio_video_features.csv
│   └── ...
├── raw_media_part_2
│   └── ...
│
│
FER file
--------
Every dialog has its own:

    audio_video_features.csv

The FER file contains:

    timedelta
    user_id
    prob_face_anger
    prob_face_contempt
    prob_face_disgust
    prob_face_fear
    prob_face_happiness
    prob_face_neutral
    prob_face_sadness
    prob_face_surprise
    ...

Important:
    timedelta is formatted as HH:MM:SS
    e.g. "00:00:03"

    user_id identifies the participant.

Transcript
----------
Each dialog has its own:

    transcript/transcript_cliffhanger.csv

The transcript contains:

    turn_id
    speaker
    start
    stop
    utterance
    questions
    end_question
    overlap

Transcript timestamps are seconds.

Analysis order
--------------

1. Load transcripts
2. Remove pre-dialog material
3. Remove pure backchannels
4. Recompute conversation position
5. Load FER
6. Match FER observations to turns using [start, stop)
7. Aggregate FER probabilities over each turn
8. Prepare one identical model dataset for H1 and H2
9. Fit H1
10. Fit H2

Models
------

H1:
    Continuous FER probabilities.

    Neutral is omitted because the eight emotion probabilities sum to 1.
    The coefficients therefore represent changes in the outcome associated
    with the respective non-neutral emotion probability relative to the
    implicit neutral component.

H2:
    Dominant facial emotion.

    Neutral is the explicit reference category.

Dependent variable:
    log(turn duration)

Random effects:
    dialog
    speaker

Fixed-effect controls:
    conversation_position
    questions
    end_question
    overlap

Important:
    The script does NOT silently fall back to OLS if the mixed model fails
    to converge. Several optimizers are tried. If none converges, the
    script stops and reports the problem.