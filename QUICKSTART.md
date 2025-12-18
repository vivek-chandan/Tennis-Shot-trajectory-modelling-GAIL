# Quick Start Guide

This is a condensed guide to get you started quickly. For detailed explanations, see [WORKFLOW.md](WORKFLOW.md).

## 1. Setup (5 minutes)

```bash
# Create environment
conda create -n tennis_gail310 python=3.10
conda activate tennis_gail310

# Install dependencies
pip install -r requirements.txt

# Login to Weights & Biases (for experiment tracking)
wandb login
```

## 2. Configure (2 minutes)

Edit `config/training_config.yaml`:
- Set `data_path` to your `ball_data.csv` location
- Set `save_dir` for model checkpoints

**Note**: The repository includes `ball_data.csv` with pre-processed trajectories.

## 3. Train Model (2-4 hours)

```bash
python main.py
```

**What happens:**
- Loads expert trajectories from CSV
- Trains GAIL model (Policy + Discriminator + TRPO)
- Saves checkpoints every 100 epochs
- Logs metrics to WandB dashboard
- Saves best model to `save_dir/best_model.pt`

**Monitor training:**
- Check WandB dashboard for real-time metrics
- Watch console for loss values
- Best model auto-saved when validation improves

## 4. Run Inference (10 seconds)

```bash
python infer.py
```

**Output:**
- `trajectory_prediction.png` - Visualization of predicted vs actual trajectory
- Console metrics showing prediction accuracy

## Understanding the Project

### What is GAIL?
**Generative Adversarial Imitation Learning** - learns to imitate expert behavior without explicit rewards.

```
Expert Data → Discriminator (Real/Fake?) → Policy (Learn to fool Discriminator)
                    ↓
            Reward Signal → TRPO → Safe Policy Updates
```

### Key Components

1. **Policy Network** (`model.py`)
   - Input: Ball state [position, velocity, distance, direction]
   - Output: Action [Δx, Δy] to change velocity
   - Architecture: 2-layer MLP with LayerNorm

2. **Discriminator** (`model.py`)
   - Input: State + Action pair
   - Output: Probability (1=expert, 0=policy)
   - Provides learning signal for policy

3. **TRPO** (`src/trpo.py`)
   - Trust Region Policy Optimization
   - Safe policy updates with KL-divergence constraint
   - Prevents catastrophic performance drops

4. **Environment** (`utils/env.py`)
   - Physics simulation (gravity, air drag)
   - Tennis court boundaries
   - Reward based on shot quality

### Data Flow

```
CSV File (ball_data.csv)
    ↓
TennisTrajectoryDataset (normalize, create sequences)
    ↓
DataLoader (batch, split train/val/test)
    ↓
Training Loop:
    - Expert batch from DataLoader
    - Policy trajectories from Environment
    - Train Discriminator
    - Update Policy with TRPO
    ↓
Saved Model (best_model.pt)
    ↓
Inference (generate predictions)
    ↓
Trajectory Visualization
```

## Pre-trained Model

Download pre-trained weights from:
[Google Drive - Model Weights](https://drive.google.com/drive/folders/10Pmu1nzGWAoue5W1C-eEd5q11JWEs37m?usp=sharing)

Place `best_model.pt` in your `save_dir` to skip training.

## Common Commands

```bash
# Train from scratch
python main.py

# Run inference with trained model
python infer.py

# Check training logs
# Go to wandb.ai and find your project "tennis-gail"

# View model checkpoints
ls chkpt/  # or your save_dir
```

## Hyperparameters (in config/training_config.yaml)

- `epochs: 300` - Training iterations
- `batch_size: 64` - Expert samples per update
- `num_trajectories: 30` - Policy rollouts per epoch
- `max_episode_length: 100` - Max steps per trajectory
- `lr: 0.00001` - Discriminator learning rate

## Expected Results

**Training Metrics:**
- Discriminator Loss: Should decrease from ~0.8 to ~0.3
- Discriminator Accuracy: Should stabilize around 0.7-0.8
- TRPO Success Rate: Should be 50-80%

**Inference:**
- Final Position Error: ~40-60 pixels (depending on trajectory length)
- Smooth, realistic trajectories
- Respects physics constraints

## Troubleshooting

**Issue**: "wandb not logged in"
```bash
wandb login
# Enter your API key from wandb.ai
```

**Issue**: "File not found: ball_data.csv"
- Check `data_path` in `config/training_config.yaml`
- Ensure path is absolute or relative to script location

**Issue**: Training very slow
- Reduce `num_trajectories` (e.g., 30 → 10)
- Use GPU if available (automatic detection)

**Issue**: Out of memory
- Reduce `batch_size` (e.g., 64 → 32)
- Reduce `num_trajectories`

## Next Steps

1. ✅ Complete this quick start
2. 📖 Read [WORKFLOW.md](WORKFLOW.md) for detailed explanations
3. 🔧 Experiment with hyperparameters
4. 📊 Analyze results in WandB dashboard
5. 🎯 Try inference on different trajectory samples

## File Structure

```
Tennis-Shot-trajectory-modelling-GAIL/
├── main.py              # Training script (START HERE)
├── infer.py             # Inference script
├── model.py             # PolicyNetwork & Discriminator
├── config/
│   └── training_config.yaml  # Configuration
├── dataset/
│   ├── data_prep.py     # TennisTrajectoryDataset
│   └── dataloader.py    # DataLoader setup
├── src/
│   └── trpo.py          # TRPO algorithm
├── utils/
│   ├── env.py           # TennisEnvironment (physics)
│   └── constants.py     # Physics constants
├── ball_data.csv        # Training data
├── requirements.txt     # Dependencies
├── QUICKSTART.md        # This file
└── WORKFLOW.md          # Detailed documentation
```

## Key Concepts Summary

**GAIL**: Learn by imitating expert demonstrations
**TRPO**: Safe reinforcement learning updates
**Physics Simulation**: Realistic ball dynamics
**Adversarial Training**: Discriminator guides policy learning

For complete details on each component, algorithms, and architecture, see [WORKFLOW.md](WORKFLOW.md).

---

**Ready to start?** Run `python main.py` and watch your model learn! 🎾
