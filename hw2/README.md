# AI Capstone HW2

NYCU AI Capstone HW2

Spec: [Google Docs](https://docs.google.com/document/d/1WmlwZPoDuV3adccKl7DBUhWRT2XjSQB81wIH2tBfEIM/edit?pli=1&tab=t.0)

## 1. Preparation
The replica dataset, you can use the same one in `hw0`.

### 2. Environment Setup (Critical)
To avoid **Segmentation Faults** and library conflicts (especially on **macOS M4** and **Linux**):

- **Python Version**: 3.9 or 3.10 is recommended.
- **NumPy Version**: You **MUST** use `numpy==1.26.4`. (Do NOT use 2.0+).
- **System Libraries**:
  - **macOS**: `brew install libomp`
  - **Linux**: `sudo apt-get install libomp-dev`

### 3. Recommended Environment Isolation
We highly recommend using two separate Conda environments to avoid **GLFW conflicts** between Habitat and Open3D:
1. **Env 1**: For data collection (`load.py`).
2. **Env 2**: For reconstruction (`reconstruct.py`).

```bash
# Install Env 2 for Reconstruction
conda create -n hw2_reconstruct python=3.9
conda activate hw2_reconstruct
pip install numpy==1.26.4 open3d scipy opencv-python pillow
```

## Phase 1: Data Collection

Use the Habitat environment to collect RGB-D images and ground truth poses.

### Command

```bash
# Example: Navigating and saving data for the first floor
python load.py
```

##Phase 2: 3D Reconstruction
Switch to the reconstruction environment before running the following commands.
⚠️ Requirements:
open3d
numpy==1.26.4
### Standard Version (Open3D ICP)
Use Open3D's built-in ICP algorithm for reconstruction.
```bash
# Reconstruct Floor 1
python reconstruct.py -f 1 -v open3d

# Reconstruct Floor 2
python reconstruct.py -f 2 -v open3d
```
### Bonus Version (Custom ICP Implementation)
If you have implemented your own ICP algorithm, use the my_icp option.
```bash
# Run reconstruction with your own ICP
python reconstruct.py -f 1 -v my_icp
```