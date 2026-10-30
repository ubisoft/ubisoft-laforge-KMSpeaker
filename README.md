##### Copyright Ubisoft(c) 2026

# KM-Speaker: Keypoint-Based Style Control for High-Quality Speech-Driven 3D Facial Animation and Dialogue Localization

## Overview

KM-Speaker is a high-fidelity speech-driven 3D facial animation model designed to tackle the following objective task in a unified manner:  

- **Example-based generation:** match the global style of a target animation (e.g., emotion, idiosyncrasies). 
- **Dubbing / Dialogue localization:** match a target facial animation while synchronizing the lip motion to new audio. 

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/ubisoft/ubisoft-laforge-kmspeaker.git
    cd ubisoft-laforge-kmspeaker
    ```

2. Create a Python 3.11 Conda environment:

    ```sh
    conda create -y --name kmspeaker python=3.11
    ```

3. Within the Conda environment, install the project's Python dependencies:

    ```sh
    conda activate kmspeaker
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```

4. Install the project:

    ```sh
    pip install -e .
    ```

5. Add the third-party DiffPoseTalk helpers:

    Our main paper results use the `PositionalEncoding` and `enc_dec_mask` functions from [DiffPoseTalk](https://github.com/DiffPoseTalk/DiffPoseTalk). Please copy them from [`models/common.py`](https://github.com/DiffPoseTalk/DiffPoseTalk/blob/main/models/common.py) into `src/kmspeaker/models/diffpose/common.py`. This file is listed in `.gitignore` to avoid redistributing DiffPoseTalk code.

    Please also cite DiffPoseTalk if you use this code.



## Setup

### Dataset format

The training dataset must match the following layout. Files are linked together by
`dataset_metadata.csv`, not by folder position:

```
data/
├── dataset_metadata.csv        # one row per clip (see columns below)
└── dataset/
    ├── audio/
    │   └── <audio_name>.wav    # one .wav file per recording
    ├── animation/
    │   └── <clip_name>.npy     # one file per clip, shape (n_frames, n_vertices, 3)
    └── template/
        ├── <identity>.obj      # one neutral-pose mesh per identity
        └── template_mesh.obj   # generic mesh, used only by create_region_mask.py
```

Each row of `dataset_metadata.csv` names the `audio_name`, `clip_name`, and `identity` for one
clip, and the scripts look up `audio/<audio_name>.wav`, `animation/<clip_name>.npy`, and
`template/<identity>.obj` from those values. Prepare the CSV with at least these columns, and save
it as `data/dataset_metadata.csv`:

| column       | meaning                                                              |
|--------------|-----------------------------------------------------------------------|
| `clip_name`  | matches `animation/<clip_name>.npy`                                  |
| `audio_name` | matches `audio/<audio_name>.wav`                                     |
| `identity`   | matches `template/<identity>.obj`; the neutral template identity for that row |

`template/template_mesh.obj` is a separate, single generic mesh used to compute the
disentanglement region masks (see below).

### Dataset preprocessing  

#### 1. Audio features — `create_cached_audio_dataset.py`

Resamples audio file referenced in the metadata to 16kHz and extracts per-frame hidden-state
features with a Hugging Face audio model (`facebook/w2v-bert-2.0` default).
Output is a single pickle mapping `audio_name -> hidden_states` tensor.

```sh
python -m kmspeaker.data_preprocessing.create_cached_audio_dataset \
  --audio_data_path data/dataset/audio \
  --metadata_file data/dataset_metadata.csv \
  --output_file data/preprocessed_data/dataset_w2vbert_hidden_states.pkl \
  --model_name facebook/w2v-bert-2.0 \
  --cache_dir data/audio_model
```

#### 2. Facial motion PCA — `create_multiperson_pca.py`

Fits a PCA over all clips and caches each clip's PCA coefficients:

```sh
python -m kmspeaker.data_preprocessing.create_multiperson_pca \
  --metadata_csv data/dataset_metadata.csv \
  --base_animation_path data/dataset/animation \
  --base_template_path data/dataset/template \
  --save_path_pca data/preprocessed_data/pca.pkl \
  --save_path_dataset_pca_coefficients data/preprocessed_data/dataset_pca_coefficients.pkl \
  --n_components 128
```

Arguments:

- `--save_path_pca`: output path for the fitted PCA (mean, components, explained variance).
- `--save_path_dataset_pca_coefficients`: output path for the cached PCA coefficients of each dataset clip.

#### 3. Upper/lower face keypoints — `create_multiperson_keypoints.py`

Caches upper- and lower-face keypoints for the dataset, along with their statistics (`<name>_statistics.pkl`):

```sh
python -m kmspeaker.data_preprocessing.create_multiperson_keypoints \
  --metadata data/dataset_metadata.csv \
  --base_animation_path data/dataset/animation \
  --base_template_path data/dataset/template \
  --source_keypoints_indices_pkl data/dummy_upper_and_lower_face_keypoints.pkl \
  --save_path_keypoints_upper data/preprocessed_data/dataset_keypoints_upper.pkl \
  --save_path_keypoints_lower data/preprocessed_data/dataset_keypoints_lower.pkl
```

Arguments:

- `--source_keypoints_indices_pkl`: pickle containing a dict of vertex indices, structured as below. Defaults to the provided dummy file `data/dummy_upper_and_lower_face_keypoints.pkl` (indices `0..9` and `10..19`, only meant for testing the pipeline).

```python
{
    "upper_face_indices": [...],  # vertex indices belonging to the upper-face region
    "lower_face_indices": [...],  # vertex indices belonging to the lower-face region
}
```

#### 4. Disentanglement region masks — `create_region_mask.py`

Computes the two soft per-vertex masks used for disentanglement (paper Supplementary Sec. B.3):
`audio_mask` (the paper's M_lower, the region driven by the audio) and `keypoint_mask`
(M_upper = 1 - M_lower, the region driven by the keypoints). The script is run once on a
generic neutral mesh (defaults to `data/dataset/template/template_mesh.obj`), and saves
`{mask_key: {"audio_mask": ..., "keypoint_mask": ...}}`.

```sh
python -m kmspeaker.data_preprocessing.create_region_mask \
  --region_indices_dict_path data/dummy_region_indices_dict.pkl \
  --base_mesh_path data/dataset/template/template_mesh.obj \
  --output_path data/preprocessed_data/disentanglement_region_masks.pkl \
  --beta 1.0 \
  --mask_key mask_beta_1.0
```

Arguments:

- `--region_indices_dict_path`: pickle containing a dict of vertex-index regions, structured as below. Defaults to the provided dummy file `data/dummy_region_indices_dict.pkl` (arbitrary regions, only meant for testing the pipeline).
- `--base_mesh_path`: generic neutral mesh the vertex distances are computed on.
- `--beta`: decay rate of the distance falloff between the two regions.
- `--mask_key`: top-level key the masks are saved under. Training reads the masks under
  `audio_keypoint_disentanglement.audio_type` / `keypoint_type` in `conf/train_km_speaker.yaml`
  (default `mask_beta_1.0`), so the two must match.

```python
{
    "front_face_vertices": [...],    # all front-of-face vertices considered by the distance falloff
    "eyes_forehead_region": [...],   # hard upper region (audio_mask = 0, keypoint_mask = 1)
    "mouth_jaw_region": [...],       # hard lower region (audio_mask = 1, keypoint_mask = 0)
    "upper_back_face": [...],        # back-of-head vertices treated as upper (audio_mask = 0)
    "lower_back_face": [...],        # back-of-head vertices treated as lower (audio_mask = 1)
}
```


#### Sanity check

Validate the structure of the preprocessed files before training. With the defaults used above, running
all four preprocessing scripts leaves `data/` looking like this:

```
data/
├── dataset_metadata.csv                      # metadata CSV, one row per clip
├── dataset/
│   ├── audio/ ..
│   ├── animation/ ..
│   └── template/ ..
└── preprocessed_data/
    ├── dataset_w2vbert_hidden_states.pkl     # create_cached_audio_dataset.py: audio_name -> hidden states
    ├── pca.pkl                               # create_multiperson_pca.py:      fitted PCA model
    ├── dataset_pca_coefficients.pkl          # create_multiperson_pca.py:      clip_name -> PCA coefficients
    ├── dataset_keypoints_upper.pkl           # create_multiperson_keypoints.py: clip_name -> upper-face keypoints
    ├── dataset_keypoints_upper_statistics.pkl
    ├── dataset_keypoints_lower.pkl           # create_multiperson_keypoints.py: clip_name -> lower-face keypoints
    ├── dataset_keypoints_lower_statistics.pkl
    └── disentanglement_region_masks.pkl      # create_region_mask.py:           mask_key -> {audio_mask, keypoint_mask}
```

Before launching training, check that all eight preprocessed files above exist under `data/preprocessed_data/`. 

## Training

Train with `python -m kmspeaker.train`

Notes: default config expects `data/dataset_metadata.csv` and preprocessed_data files. 

Checkpoints and logs are saved to `trained_models/<date>/<time>/` by default. 

## Inference

Run with `python -m kmspeaker.inference.sampler`. The script drives a trained checkpoint from a `.wav`
file and a neutral template mesh, and supports two mutually exclusive prediction modes selected with
`--mode`. Whichever mode you pick determines which of the two target-sequence flags is required — passing
the wrong one for the selected mode raises an assertion error.

Arguments (both modes):

- `--checkpoint_path`: path to the trained checkpoint (required).
- `--wav_path`: input speech audio driving the animation (required).
- `--target_neutral`: neutral-pose `.obj` template used to condition/normalize the target sequence.
- `--decoding_neutral`: neutral-pose `.obj` template added back after decoding. Defaults to `--target_neutral`.
- `--output_folder`: directory the output `.npy` mesh sequence is saved to (required).
- `--output_name`: output filename. Defaults to a name derived from `--wav_path`.
- `--mode`: `dubbing` or `example_based` (required), see below.
- `--cfg_guidance_scale`: classifier-free guidance scale. Defaults to `1.0` for `dubbing`, forced to `0.0` for `example_based`. Not normally set manually.
- `--target_animation_fps`: FPS of the input target/style sequence, if not already 60.
- `--scale`, `--shift`: scale/shift applied to the target sequence before conditioning.
- `--seed`, `--n_steps`, `--method`: sampler noise seed, number of ODE solver steps, and integration method.

### Dubbing

Synchronizes lip motion to the input audio while matching a target facial performance. The target sequence's own keypoints drive both the frame-wise lip motion and the global style. There is no separate style input in this mode.


```sh
python -m kmspeaker.inference.sampler \
  --checkpoint_path <checkpoint.pth> \
  --wav_path <audio.wav> \
  --target_neutral <neutral.obj> \
  --output_folder <output_dir> \
  --mode dubbing \
  --dubbing_target_sequence <target_performance.npy>
```

Arguments:

- `--dubbing_target_sequence`: path to the frame-aligned target performance `.npy` (required for this mode).
- `--ignore_global_style`: if set, zeroes the global style signal only. Lip motion is unaffected.


### Example-based generation

Matches the global style (emotion, idiosyncrasies) of a reference clip while generating lip motion purely
from the input audio — no frame-wise driving signal is used.


```sh
python -m kmspeaker.inference.sampler \
  --checkpoint_path <checkpoint.pth> \
  --wav_path <audio.wav> \
  --target_neutral <neutral.obj> \
  --output_folder <output_dir> \
  --mode example_based \
  --style_reference_sequence <style_reference.npy>
```

Arguments:

- `--style_reference_sequence`: path to the style reference clip `.npy` (required for this mode).
- `--example_style_strategy`: how style is derived from the reference clip.
  - `first_window` (default): the reference's first `sequence_length` frames are encoded once and reused as the style for the whole generation.
  - `previous_window`: the style follows the reference over time, each output window using the `sequence_length` reference frames immediately before it (windows starting before frame `sequence_length` use the first window instead). The reference is usually matching the audio for that setting, and must be at least as long as the output. If the output fits in a single window (`sequence_length` frames), it raises an error. Frames before `sequence_length` are styled by themselves, so only evaluate from frame `sequence_length` onward.


## TexTalk4D evaluation

Out-of-domain results in the paper's supplementary material are computed on the test speakers (identities `ID063` to `ID072`) of the TexTalk4D dataset ([TexTalker](https://github.com/xuanchenli/TexTalk),
CC BY-NC 4.0): Download the data from their repository.
For each identity we use:

- the ground-truth sequence: every per-frame `Models/<frame>/face_aligned.obj`, stacked in frame order into an array
  of shape (n_frames, 8280, 3), at 30 fps, in the native TexTalk4D topology;
- the neutral: the `face_aligned.obj` of the first frame;
- the audio: `Records/enhanced_vocal.wav`.

`data/textalker4d_metric_regions.pkl` provides the vertex regions for LVE, MSE and FDD on this topology. It is a dict
of vertex indices into the 8280 TexTalk4D vertices:

```python
{
    "lve_lip_vertices": [...],         # 387 lip vertices, used for LVE
    "fdd_upper_face_vertices": [...],  # 1602 upper-face vertices, used for FDD
    "mse_excluded_vertices": [...],    # 4081 eye, mouth-interior and back-of-head vertices, left out of MSE
}
```

## Citation 

```bibtex
@article{josi2026km,
  title={KM-Speaker: Keypoint-Based Style Control for High-Quality Speech-Driven 3D Facial Animation and Dialogue Localization},
  author={Josi, Arthur and Got, Emeline and Dib, Abdallah and Hafemann, Luiz Gustavo and Cruz, Rafael MO},
  journal={arXiv preprint arXiv:2606.28568},
  year={2026}
}
```

##### Copyright Ubisoft(c) 2026
