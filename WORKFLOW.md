# Tennis Shot Trajectory Modelling with GAIL - Complete Workflow

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Architecture Overview](#architecture-overview)
4. [Dataset Collection & Preparation](#dataset-collection--preparation)
5. [Data Processing Pipeline](#data-processing-pipeline)
6. [Model Components](#model-components)
7. [Training Workflow](#training-workflow)
8. [Inference Workflow](#inference-workflow)
9. [Configuration System](#configuration-system)
10. [Complete Execution Flow](#complete-execution-flow)

---

## Project Overview

This project implements **Generative Adversarial Imitation Learning (GAIL)** for modeling tennis ball trajectories. GAIL is an inverse reinforcement learning method that learns to imitate expert behavior by using:
- **Policy Network**: Generates ball trajectories (actions)
- **Discriminator**: Distinguishes expert trajectories from policy-generated ones
- **TRPO (Trust Region Policy Optimization)**: Updates the policy network safely

### Key Objectives
- Learn realistic tennis ball trajectories from expert data
- Model physics-based ball movement (gravity, drag, air resistance)
- Predict future ball positions given current state
- Generate human-like shot patterns

---

## Prerequisites & Setup

### Step 1: Environment Setup
```bash
# Create conda environment with Python 3.10
conda create -n tennis_gail310 python=3.10
conda activate tennis_gail310
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required Packages:**
- `torch>=2.0.0` - Deep learning framework
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `wandb>=0.15.0` - Experiment tracking
- `scipy>=1.7.0` - Scientific computing
- `matplotlib>=3.5.0` - Visualization

### Step 3: Configure Paths
Edit `config/training_config.yaml` to set:
- `data_path`: Path to your ball_data.csv
- `save_dir`: Directory for saving model checkpoints

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GAIL Training Loop                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │   Expert     │      │   Policy     │                     │
│  │   Data       │      │   Network    │                     │
│  │  (ball_data) │      │  (Generator) │                     │
│  └──────┬───────┘      └──────┬───────┘                    │
│         │                     │                              │
│         │ Real Trajectories   │ Generated Trajectories       │
│         │                     │                              │
│         └─────────┬───────────┘                             │
│                   │                                          │
│                   ▼                                          │
│          ┌────────────────┐                                 │
│          │ Discriminator  │                                 │
│          │ (Real vs Fake) │                                 │
│          └────────┬───────┘                                 │
│                   │                                          │
│                   │ Reward Signal                            │
│                   ▼                                          │
│          ┌────────────────┐                                 │
│          │      TRPO      │                                 │
│          │ (Policy Update)│                                 │
│          └────────┬───────┘                                 │
│                   │                                          │
│                   └──► Update Policy Network                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset Collection & Preparation

### Phase 1: Video Processing (dataset_collection_preparation/)

The project includes tools for collecting trajectory data from tennis videos:

#### Key Components:
1. **Tennis Tracking Module** (`dataset_prep/tennis-tracking/`)
   - `predict_video.py`: Video processing main script
   - `detection.py`: Ball and player detection using YOLO
   - `TrackPlayers/trackplayers.py`: Player tracking algorithm
   - `Models/tracknet.py`: TrackNet model for ball tracking
   - `court_detector.py`: Court boundary detection
   - `court_reference.py`: Court coordinate system
   - `sort.py`: SORT algorithm for multi-object tracking

#### Workflow:
```
Video Input → Frame Extraction → Ball Detection (YOLO/TrackNet)
                                      ↓
                              Player Detection
                                      ↓
                              Multi-Object Tracking (SORT)
                                      ↓
                              Court Coordinate Mapping
                                      ↓
                              Trajectory Data (CSV)
```

### Phase 2: Output Format
The processed data is saved in `ball_data.csv` with columns:
- `frame`: Frame number
- `ball_x`: Ball X position (pixels)
- `ball_y`: Ball Y position (pixels)
- `velocity_x`: Velocity in X direction
- `velocity_y`: Velocity in Y direction
- `shot_type`: Type of shot (0 or 1 for direction)
- `distance_travelled`: Distance since last hit

---

## Data Processing Pipeline

### Step 1: Data Loading (`dataset/data_prep.py`)

**TennisTrajectoryDataset Class:**
```python
# Input: ball_data.csv
# Output: State-action sequence pairs
```

**Processing Steps:**

1. **Load CSV Data**
   - Reads ball_data.csv
   - Skips header row
   - Maps columns to features

2. **Normalization**
   ```python
   # For each feature: ball_x, ball_y, velocity_x, velocity_y, distance_travelled
   normalized_value = (value - mean) / std
   
   # Scalers are stored for later denormalization
   scalers[feature] = (mean, std)
   ```

3. **Sequence Creation**
   - Creates sliding windows of length 20 frames
   - Converts to state-action pairs:
     - **State**: [ball_x, ball_y, velocity_x, velocity_y, distance_travelled, shot_type]
     - **Action**: [Δball_x, Δball_y] (change in position)

4. **Output Format**
   ```python
   {
     'states': torch.FloatTensor([seq_length-1, 6]),  # 19 states
     'actions': torch.FloatTensor([seq_length-1, 2])  # 19 actions
   }
   ```

### Step 2: DataLoader Creation (`dataset/dataloader.py`)

**Split Strategy:**
- Training: 80% of sequences
- Validation: 10% of sequences
- Test: 10% of sequences

**Batch Processing:**
- Default batch size: 64
- Shuffling: True for training, False for validation/test

---

## Model Components

### 1. Policy Network (`model.py` - PolicyNetwork)

**Purpose:** Generates actions (ball movements) given current state

**Architecture:**
```
Input (6 dims): [ball_x, ball_y, vx, vy, distance, direction]
    ↓
Linear(6 → 256) + LayerNorm + ReLU
    ↓
Linear(256 → 256) + LayerNorm + ReLU
    ↓
    ├─→ Mean Head: Linear(256 → 2)
    └─→ Std Head: Learnable parameter (2 dims)
    ↓
Gaussian Distribution(mean, std)
    ↓
Output (2 dims): [Δx, Δy] (action)
```

**Key Features:**
- Layer normalization for training stability
- Gaussian policy for continuous actions
- Outputs both mean and standard deviation
- Stochastic action sampling during training

**Forward Pass:**
```python
mean, std = policy(state)           # Get distribution parameters
dist = Normal(mean, std)            # Create Gaussian distribution
action = dist.sample()              # Sample action
log_prob = dist.log_prob(action)    # For TRPO update
```

### 2. Discriminator (`model.py` - Discriminator)

**Purpose:** Distinguish expert trajectories from policy-generated ones

**Architecture:**
```
Input (8 dims): [state(6) + action(2)]
    ↓
Linear(8 → 128) + ReLU
    ↓
Linear(128 → 128) + ReLU
    ↓
Linear(128 → 1) + Sigmoid
    ↓
Output (1 dim): Probability [0, 1]
    - Close to 1: Expert trajectory
    - Close to 0: Policy trajectory
```

**Loss Function:**
```python
# Binary cross-entropy style loss
disc_loss = -(log(D(expert)) + log(1 - D(policy)))
```

### 3. Tennis Environment (`utils/env.py`)

**Purpose:** Physics-based simulation of tennis ball movement

**State Representation:**
```python
state = [
    ball_position_x,      # X coordinate on court
    ball_position_y,      # Y coordinate on court
    ball_velocity_x,      # Velocity in X direction
    ball_velocity_y,      # Velocity in Y direction
    distance_since_hit,   # Distance traveled since last contact
    direction            # 0 (left) or 1 (right)
]
```

**Physics Simulation:**
```python
# Forces acting on ball:
1. Gravity: F_g = -GRAVITY (downward)
2. Air Drag: F_d = -0.5 * ρ * C_d * A * |v| * v

# Update equations (Euler integration):
acceleration = [0, -GRAVITY] + drag/mass
velocity += acceleration * dt
position += velocity * dt
```

**Constants (utils/constants.py):**
- `COURT_DIMENSIONS`: (1200, 1200) pixels
- `GRAVITY`: 9.81 m/s²
- `DRAG_COEFFICIENT`: 0.47
- `AIR_DENSITY`: 1.225 kg/m³
- `BALL_RADIUS`: 0.067 m
- `BALL_MASS`: 0.0577 kg
- `DT`: 0.01 seconds (time step)

**Reward Function:**
```python
# Exponential reward based on distance to target
target = [court_length/2, court_width]
distance = euclidean_distance(ball_position, target)
reward = exp(-distance/5.0)
```

**Episode Termination:**
- Ball goes out of court boundaries
- Ball Y position becomes negative
- Maximum steps reached

### 4. TRPO Algorithm (`src/trpo.py`)

**Purpose:** Safe policy updates with KL-divergence constraint

**Key Hyperparameters:**
- `max_kl`: 0.01 (Maximum KL divergence between old and new policy)
- `damping`: 0.1 (Fisher vector product damping)

**Update Steps:**

1. **Store Old Policy**
   ```python
   old_policy = copy(current_policy)
   ```

2. **Calculate Advantages**
   ```python
   # Use discriminator as reward signal
   advantages = log(discriminator(states, actions))
   ```

3. **Compute Surrogate Loss**
   ```python
   ratio = exp(new_log_prob - old_log_prob)
   loss = mean(ratio * advantages)
   ```

4. **Natural Gradient Calculation**
   - Compute policy gradient
   - Calculate Fisher Information Matrix
   - Solve: F⁻¹ * gradient using Conjugate Gradient

5. **Line Search**
   ```python
   # Try step sizes: [1.0, 0.5, 0.25, 0.125, ...]
   for α in [0.5^i for i in range(10)]:
       new_params = old_params + α * natural_gradient
       if improvement AND kl <= max_kl:
           accept_update()
           break
   ```

6. **Backtracking**
   - If no acceptable step found, reject update
   - Keep old policy parameters

---

## Training Workflow

### Complete Training Process (`main.py`)

#### Initialization Phase

```python
# 1. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load configuration
config = load_config()  # From config/training_config.yaml

# 3. Initialize Weights & Biases tracking
wandb.init(project="tennis-gail", config=config)

# 4. Create data loaders
train_loader, val_loader, test_loader = create_dataloader(
    config['data_path'], 
    batch_size=config['batch_size']
)

# 5. Initialize environment
env = TennisEnvironment()

# 6. Create models
policy = PolicyNetwork(state_dim=6, action_dim=2).to(device)
discriminator = Discriminator(state_dim=6, action_dim=2).to(device)

# 7. Initialize optimizer and TRPO
disc_optimizer = Adam(discriminator.parameters(), lr=config['lr'])
trpo = TRPO(policy, env, discriminator)
```

#### Training Loop (Per Epoch)

```
For epoch in range(epochs):
    ┌─────────────────────────────────────┐
    │      TRAINING PHASE                 │
    ├─────────────────────────────────────┤
    │                                     │
    │ 1. Get expert batch from training   │
    │    data loader                      │
    │    expert_states, expert_actions    │
    │                                     │
    │ 2. Collect policy trajectories      │
    │    - Run policy in environment      │
    │    - Collect num_trajectories       │
    │    - Each trajectory: states,       │
    │      actions until done             │
    │                                     │
    │ 3. Train Discriminator              │
    │    real_score = D(expert)           │
    │    fake_score = D(policy)           │
    │    loss = -[log(real) + log(1-fake)]│
    │    disc_optimizer.step()            │
    │                                     │
    │ 4. Update Policy with TRPO          │
    │    - Compute advantages from D      │
    │    - Calculate natural gradient     │
    │    - Line search for safe update    │
    │                                     │
    └─────────────────────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │     VALIDATION PHASE                │
    ├─────────────────────────────────────┤
    │                                     │
    │ 1. Get validation batch             │
    │                                     │
    │ 2. Collect validation trajectories  │
    │                                     │
    │ 3. Calculate validation metrics:    │
    │    - Discriminator loss             │
    │    - Expert accuracy (D > 0.5)      │
    │    - Policy accuracy (D < 0.5)      │
    │    - Total accuracy                 │
    │                                     │
    └─────────────────────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │        LOGGING                      │
    ├─────────────────────────────────────┤
    │                                     │
    │ wandb.log({                         │
    │   "train/disc_loss": ...,           │
    │   "train/d_total_acc": ...,         │
    │   "val/disc_loss": ...,             │
    │   "val/d_total_acc": ...,           │
    │   "trpo_success": ...               │
    │ })                                  │
    │                                     │
    └─────────────────────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │      CHECKPOINTING                  │
    ├─────────────────────────────────────┤
    │                                     │
    │ if epoch % save_interval == 0:      │
    │   Save checkpoint                   │
    │                                     │
    │ if val_loss < best_loss:            │
    │   Save best model                   │
    │   Update best_loss                  │
    │                                     │
    └─────────────────────────────────────┘
```

#### Trajectory Collection Function

```python
def collect_policy_trajectories(policy, env, num_trajs, device):
    """
    Rollout policy in environment to collect trajectories
    """
    trajectories = []
    
    for _ in range(num_trajs):
        state = env.reset()
        episode_states = []
        episode_actions = []
        done = False
        
        while not done:
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, _ = policy.act(state_tensor)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            episode_states.append(state)
            episode_actions.append(action)
            
            state = next_state
        
        # Convert to tensors
        trajectories.append({
            'states': torch.FloatTensor(episode_states).to(device),
            'actions': torch.FloatTensor(episode_actions).to(device)
        })
    
    return trajectories
```

#### Final Test Evaluation

```python
# After training completes:
policy.eval()
discriminator.eval()

# Get test batch
test_states, test_actions = next(iter(test_loader))

# Collect test trajectories
test_trajectories = collect_policy_trajectories(policy, env, num_trajectories)

# Calculate test metrics
test_disc_loss = ...
test_accuracy = ...

# Log to wandb
wandb.log({"test/disc_loss": ..., "test/d_total_acc": ...})

# Save loss plot
plot_discriminator_loss(disc_losses, save_path="loss_curve.png")
wandb.finish()
```

### Training Metrics Tracked

**Training Metrics:**
- `train/disc_loss`: Discriminator loss on training data
- `train/d_expert_acc`: Accuracy on identifying expert trajectories
- `train/d_policy_acc`: Accuracy on identifying policy trajectories
- `train/d_total_acc`: Average of both accuracies

**Validation Metrics:**
- `val/disc_loss`: Discriminator loss on validation data
- `val/d_total_acc`: Validation accuracy

**Test Metrics:**
- `test/disc_loss`: Final test loss
- `test/d_total_acc`: Final test accuracy

**TRPO Metrics:**
- `trpo_success`: Boolean indicating if TRPO update was accepted

---

## Inference Workflow

### Complete Inference Process (`infer.py`)

#### Step 1: Load Trained Components

```python
def load_inference_components(model_path, data_path):
    """
    Load trained models and data scalers
    """
    # 1. Initialize environment
    env = TennisEnvironment()
    
    # 2. Load dataset to get scalers
    dataset = TennisTrajectoryDataset(data_path)
    scalers = dataset.scalers
    
    # 3. Create model instances
    policy = PolicyNetwork(state_dim=6, action_dim=2)
    discriminator = Discriminator(state_dim=6, action_dim=2)
    
    # 4. Load checkpoint
    checkpoint = torch.load(f"{model_path}/best_model.pt")
    policy.load_state_dict(checkpoint['policy_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # 5. Set evaluation mode
    policy.eval()
    discriminator.eval()
    
    return env, policy, scalers
```

#### Step 2: Initialize Environment from Data

```python
def initialize_environment_from_data(env, scalers, sample_idx=10):
    """
    Set environment to a specific state from dataset
    """
    # 1. Get sample from dataset
    dataset = TennisTrajectoryDataset(data_path)
    sample = dataset[sample_idx]
    initial_state_normalized = sample['states'][0].numpy()
    
    # 2. Denormalize state values
    def denormalize(col, value):
        mean, std = scalers[col]
        return value * std + mean
    
    # 3. Set environment state
    env.ball_position = np.array([
        denormalize('ball_x', initial_state_normalized[0]),
        denormalize('ball_y', initial_state_normalized[1])
    ])
    
    env.ball_velocity = np.array([
        denormalize('velocity_x', initial_state_normalized[2]),
        denormalize('velocity_y', initial_state_normalized[3])
    ])
    
    env.distance_since_last_hit = denormalize(
        'distance_travelled', 
        initial_state_normalized[4]
    )
    
    env.direction = 1 if env.ball_velocity[0] > 0 else 0
    
    return env._get_state()
```

#### Step 3: Generate Trajectory

```python
def generate_trajectory(env, policy, scalers, max_steps=10000):
    """
    Generate predicted trajectory using trained policy
    """
    state = env._get_state()
    actual_positions = [env.ball_position.copy()]
    predicted_positions = []
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        # 1. Normalize current state
        normalized_state = normalize_state(state, scalers)
        
        # 2. Get action from policy
        with torch.no_grad():
            action_mean, _ = policy(torch.FloatTensor(normalized_state).unsqueeze(0))
            action = action_mean.squeeze().numpy()
        
        # 3. Store prediction
        current_pos = actual_positions[-1]
        # Scale action back to original space
        predicted_pos = current_pos + action * scalers['velocity_x'][1]
        predicted_positions.append(predicted_pos)
        
        # 4. Step environment (physics simulation)
        next_state, _, done, _ = env.step(action)
        actual_positions.append(next_state[:2])  # Store position only
        
        state = next_state
        step += 1
    
    return np.array(actual_positions), np.array(predicted_positions)
```

#### Step 4: Visualize Results

```python
def plot_trajectory(actual, predicted, save_path="trajectory_prediction.png"):
    """
    Plot actual vs predicted trajectories with court layout
    """
    plt.figure(figsize=(12, 8))
    
    # 1. Draw court boundaries
    court_length, court_width = COURT_DIMENSIONS
    plt.plot([0, court_length], [0, 0], 'k-', lw=3, label='Baseline')
    plt.plot([0, court_length], [court_width, court_width], 'k--', lw=2)
    plt.plot([court_length/2, court_length/2], [0, court_width], 'k:')
    
    # 2. Plot trajectories (subsample for clarity)
    plt.plot(actual[::1000, 0], actual[::1000, 1], 
             'g-o', label='Actual Path', markersize=8)
    plt.plot(predicted[::1000, 0], predicted[::1000, 1], 
             'b--s', label='Predicted Path', markersize=6)
    
    # 3. Draw error vectors
    for i in range(0, len(predicted), 1000):
        plt.annotate('', xy=predicted[i], xytext=actual[i],
                    arrowprops=dict(arrowstyle="->", color='r', alpha=0.5))
    
    # 4. Formatting
    plt.title("Ball Trajectory Prediction", fontsize=14)
    plt.xlabel("X Position (pixels)", fontsize=12)
    plt.ylabel("Y Position (pixels)", fontsize=12)
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

#### Step 5: Calculate Metrics

```python
# Calculate trajectory error
final_error = np.linalg.norm(actual[-1] - predicted[-1])
mean_error = np.mean([np.linalg.norm(a - p) 
                      for a, p in zip(actual, predicted)])

print(f"Final Position Error: {final_error:.2f} pixels")
print(f"Mean Trajectory Error: {mean_error:.2f} pixels")
print(f"Trajectory Length: {len(actual)} steps")
```

### Running Inference

```bash
# Execute inference script
python infer.py
```

**Output:**
- `trajectory_prediction.png`: Visualization of predicted vs actual trajectory
- Console output with error metrics
- Trajectory data arrays

---

## Configuration System

### Configuration File (`config/training_config.yaml`)

```yaml
training_config:
  # Data
  data_path: "/path/to/ball_data.csv"
  
  # Training hyperparameters
  epochs: 300                    # Number of training epochs
  batch_size: 64                 # Batch size for expert data
  num_trajectories: 30           # Number of policy rollouts per epoch
  max_episode_length: 100        # Maximum steps per episode
  
  # Learning
  lr: 0.00001                    # Discriminator learning rate
  
  # Checkpointing
  save_dir: "/path/to/checkpoints"
  save_interval: 100             # Save checkpoint every N epochs
```

### Key Hyperparameters Explained

**epochs**: Total training iterations
- Higher values → Better convergence but longer training
- Typical: 200-500 epochs

**batch_size**: Number of expert sequences per training step
- Higher → More stable gradients, more memory
- Typical: 32-128

**num_trajectories**: Policy rollouts per epoch
- Higher → Better policy gradient estimates
- Trade-off: Computational cost vs accuracy
- Typical: 10-50

**max_episode_length**: Maximum steps per trajectory
- Should match typical shot duration
- Prevents infinite loops
- Typical: 50-200

**lr (learning rate)**: Discriminator update step size
- Too high → Unstable training
- Too low → Slow convergence
- Typical: 1e-5 to 1e-3

### TRPO Hyperparameters (in code)

**max_kl**: 0.01
- Maximum KL divergence constraint
- Lower → Safer but slower updates

**damping**: 0.1
- Fisher matrix regularization
- Numerical stability for CG solver

---

## Complete Execution Flow

### From Scratch: Full Pipeline

```
┌────────────────────────────────────────────────────────┐
│                 STEP 1: SETUP                          │
├────────────────────────────────────────────────────────┤
│ 1. conda create -n tennis_gail310 python=3.10          │
│ 2. pip install -r requirements.txt                     │
│ 3. wandb login                                         │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│         STEP 2: DATA PREPARATION (Optional)            │
├────────────────────────────────────────────────────────┤
│ If collecting new data:                                │
│ 1. Record tennis match videos                         │
│ 2. cd dataset_collection_preparation/dataset_prep/     │
│    tennis-tracking                                     │
│ 3. python predict_video.py --video path/to/video.mp4  │
│ 4. Output: trajectory CSV file                        │
│                                                        │
│ Or use provided:                                       │
│ - ball_data.csv (pre-processed trajectories)          │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│         STEP 3: CONFIGURE TRAINING                     │
├────────────────────────────────────────────────────────┤
│ Edit config/training_config.yaml:                     │
│ - Set data_path to your ball_data.csv                 │
│ - Set save_dir for checkpoints                        │
│ - Adjust hyperparameters if needed                    │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│              STEP 4: TRAINING                          │
├────────────────────────────────────────────────────────┤
│ python main.py                                         │
│                                                        │
│ Training loop runs for 300 epochs:                    │
│                                                        │
│ Each epoch:                                           │
│   ├─→ Load expert batch                              │
│   ├─→ Collect 30 policy trajectories                 │
│   ├─→ Train discriminator                            │
│   ├─→ Update policy with TRPO                        │
│   ├─→ Validate on validation set                     │
│   ├─→ Log metrics to wandb                           │
│   └─→ Save checkpoint if best model                  │
│                                                        │
│ Outputs:                                              │
│ - Checkpoints in save_dir/checkpoint_epoch*.pt        │
│ - Best model: save_dir/best_model.pt                  │
│ - Loss curve: save_dir/loss_curve.png                 │
│ - Wandb logs: Online dashboard                       │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│            STEP 5: INFERENCE                           │
├────────────────────────────────────────────────────────┤
│ python infer.py                                        │
│                                                        │
│ Inference pipeline:                                    │
│   ├─→ Load best_model.pt                             │
│   ├─→ Load data scalers                              │
│   ├─→ Initialize environment from dataset sample     │
│   ├─→ Generate trajectory using trained policy       │
│   ├─→ Compare with actual (physics-simulated)        │
│   └─→ Plot and save results                          │
│                                                        │
│ Outputs:                                              │
│ - trajectory_prediction.png                           │
│ - Error metrics in console                           │
└────────────────────────────────────────────────────────┘
```

### Detailed Execution Timeline

#### Training Session (300 epochs, ~2-4 hours on GPU)

```
00:00:00 - Initialize components
         - Load config
         - Setup wandb
         - Create dataloaders (split 80/10/10)
         - Initialize policy, discriminator, TRPO

00:00:30 - Epoch 0
         - Get expert batch (64 sequences)
         - Collect 30 policy trajectories
         - Train discriminator
         - TRPO update
         - Validation
         - Log metrics
         [Train Loss: 0.8234, Val Loss: 0.7891]

00:01:00 - Epoch 1
         ...

00:10:00 - Epoch 10
         [Train Loss: 0.6123, Val Loss: 0.6001]

01:00:00 - Epoch 100 (Checkpoint saved)
         [Train Loss: 0.4567, Val Loss: 0.4523]
         ✓ New best model saved

02:00:00 - Epoch 200 (Checkpoint saved)
         [Train Loss: 0.3234, Val Loss: 0.3198]
         ✓ New best model saved

03:00:00 - Epoch 299 (Final)
         [Train Loss: 0.2891, Val Loss: 0.2856]

03:00:30 - Test evaluation
         [Test Loss: 0.2879, Test Acc: 0.8234]

03:01:00 - Save plots and finish
         - loss_curve.png saved
         - Training complete
```

#### Inference Session (~10 seconds)

```
00:00:00 - Load components
         - Load best_model.pt
         - Load data scalers
         - Initialize environment

00:00:02 - Generate trajectory
         - Sample initial state from dataset
         - Run policy for max 10000 steps or until done
         - Collect actual (physics) and predicted positions

00:00:08 - Visualize and evaluate
         - Create trajectory plot
         - Calculate error metrics
         - Save trajectory_prediction.png

00:00:10 - Complete
         Final Position Error: 45.23 pixels
         Trajectory Length: 8234 steps
```

---

## Key Insights & Best Practices

### Understanding GAIL in This Context

**Why GAIL for Tennis Trajectories?**
1. **Imitation Learning**: Learn from expert demonstrations without explicit rewards
2. **Adversarial Training**: Discriminator provides implicit reward signal
3. **Physics Integration**: Policy learns to respect physical constraints naturally
4. **Continuous Actions**: Gaussian policy handles continuous velocity changes

### Training Tips

1. **Monitor Discriminator Accuracy**
   - Should stabilize around 0.7-0.8
   - Too high → Discriminator too strong, policy can't learn
   - Too low → Discriminator too weak, no learning signal

2. **TRPO Success Rate**
   - Should be 50-80% of epochs
   - Too low → Reduce max_kl or increase num_trajectories
   - Too high → Can increase max_kl for faster learning

3. **Validation Loss**
   - Should decrease smoothly
   - Sharp increases → Reduce learning rate
   - Plateaus → May need more training data or model capacity

### Common Issues & Solutions

**Issue**: Policy generates out-of-bounds trajectories
- **Solution**: Increase num_trajectories for better exploration
- Check environment boundaries in constants.py

**Issue**: Discriminator loss oscillates
- **Solution**: Reduce learning rate
- Increase batch size for more stable gradients

**Issue**: TRPO updates always rejected
- **Solution**: Increase max_kl (e.g., 0.01 → 0.05)
- Check if advantages are reasonable magnitude

**Issue**: Training very slow
- **Solution**: Reduce num_trajectories
- Use GPU acceleration
- Reduce max_episode_length

### Model Download

Pre-trained model weights available at:
[Google Drive - Best Model Weights](https://drive.google.com/drive/folders/10Pmu1nzGWAoue5W1C-eEd5q11JWEs37m?usp=sharing)

Download and place in your `save_dir` as `best_model.pt`

---

## Summary

This project implements a complete pipeline for learning tennis ball trajectories using GAIL:

1. **Data Collection**: Video processing → Ball tracking → Trajectory extraction
2. **Data Processing**: CSV → Normalized sequences → State-action pairs
3. **Model Training**: GAIL with TRPO → Adversarial learning → Safe policy updates
4. **Inference**: Trained policy → Physics simulation → Trajectory prediction

**Key Components:**
- **PolicyNetwork**: Generates realistic ball movements
- **Discriminator**: Distinguishes expert from policy trajectories
- **TRPO**: Safe reinforcement learning updates
- **TennisEnvironment**: Physics-based simulation (gravity, drag, boundaries)

**Workflow:**
```
Expert Data → GAIL Training → Trained Policy → Trajectory Prediction
```

The system learns to generate human-like tennis shot trajectories by imitating expert demonstrations while respecting physical constraints, enabling realistic ball trajectory prediction for various applications in sports analytics and gaming.
