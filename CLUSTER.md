# ML/DL Cluster Setup Documentation

## Cluster Architecture

### Hardware Configuration
- **Master/Controller**: Windows + WSL (Ubuntu 24.04) - Development and control node
- **Worker1 (Laptop)**: Ubuntu 22.04 - Main compute node, gateway, dataset storage
- **Worker2 (Desktop)**: Ubuntu 22.04 - Secondary compute node

### Network Configuration
- **LAN Network**:
  - Worker1: `10.10.10.1` (interface: `eno1`)
  - Worker2: `10.10.10.2` (interface: `enp2s0`)
- **Internet**: Worker2 gets internet through Worker1 via NAT
- **SSH Access**:
  - WSL → Worker1: Direct
  - WSL → Worker2: ProxyJump via Worker1
  - Worker1 → Worker2: Direct

## System Components

### Python Environment
- **Python Version**: 3.12.7 (built from source)
- **Location**: `/opt/python3.12.7`
- **Virtual Environment**: `~/cluster-work/venv3.12.7`
- **Installed Packages**: numpy, pandas, matplotlib, scikit-learn, jupyter, psutil, tqdm, joblib, ray

### Directory Structure
```
~/cluster-work/
├── bin/                 # Scripts and utilities
├── datasets/            # Large datasets (>2GB)
├── node-logs/           # Logs and results
├── repos/              # Project code repositories
├── queue/              # Task queue system files
├── venv3.12.7/         # Python virtual environment
├── queue_server.py     # Task queue server
├── ray_train.py        # Ray distributed training example
├── test_job.py         # Test job script
├── test_parallel.py    # Parallel test script
├── worker_client.py    # Worker client for task queue
└── worker*.log         # Worker log files
```

## Cluster Management Scripts

### Task Queue System
Located in `~/cluster-work/bin/`:

- **cluster-run**: Run commands on both workers
- **cluster-worker**: Worker daemon for task queue
- **run-job**: Run Python jobs on cluster (parallel)
- **task-manager**: Task queue manager

### Ray Distributed System
- **Head Node**: Worker1 (`ray start --head --port=6379`)
- **Worker Node**: Worker2 (`ray start --address='10.10.10.1:6379'`)

## How to Use the Cluster

### 1. Activate Environment
Always activate the virtual environment before running any Python code:
```bash
ssh worker1
cd ~/cluster-work
source venv3.12.7/bin/activate

### 2. Start Ray Cluster (for distributed computing)
```bash
# On Worker1 (head node)
ray start --head --port=6379

# On Worker2 (worker node)
ssh worker2 "cd ~/cluster-work && source venv3.12.7/bin/activate && ray start --address='10.10.10.1:6379'"

# To stop Ray
ray stop
```

### 3. Run Distributed Python Code with Ray

#### Basic Example:
```python
import ray
import socket

# Connect to existing Ray cluster
ray.init(address='auto')

@ray.remote
def distributed_function():
    return f"Running on: {socket.gethostname()}"

# Run tasks distributed across cluster
futures = [distributed_function.remote() for _ in range(10)]
results = ray.get(futures)
print(results)
```

#### Data Parallel Training Example:
```python
import ray
import numpy as np
import torch

ray.init(address='auto')

@ray.remote(num_gpus=0)  # Set num_gpus=1 if using GPUs
def train_on_chunk(chunk_data, hyperparams):
    """
    Train model on a data chunk
    This function runs on whichever worker is available
    """
    # Your training code here
    model = YourModel()
    accuracy = model.train(chunk_data, hyperparams)
    return accuracy

# Load and split dataset on head node
dataset = load_your_dataset()  # Load once
chunks = np.array_split(dataset, 10)  # Split into 10 chunks

# Distribute training across cluster
futures = [train_on_chunk.remote(chunk, hyperparams) for chunk in chunks]
results = ray.get(futures)

# Aggregate results
final_accuracy = np.mean(results)
print(f"Final accuracy: {final_accuracy}")
```

#### Hyperparameter Tuning:
```python
import ray

ray.init(address='auto')

@ray.remote
def train_with_params(lr, batch_size, dropout):
    # Train model with given hyperparameters
    accuracy = train_model(lr, batch_size, dropout)
    return {'lr': lr, 'batch_size': batch_size, 'accuracy': accuracy}

# Define hyperparameter combinations
param_grid = [
    {'lr': 0.001, 'batch_size': 32, 'dropout': 0.2},
    {'lr': 0.001, 'batch_size': 64, 'dropout': 0.2},
    {'lr': 0.01, 'batch_size': 32, 'dropout': 0.2},
    {'lr': 0.01, 'batch_size': 64, 'dropout': 0.2},
]

# Run all combinations in parallel
futures = [train_with_params.remote(**params) for params in param_grid]
results = ray.get(futures)

# Find best hyperparameters
best_result = max(results, key=lambda x: x['accuracy'])
print(f"Best hyperparameters: {best_result}")
```

### 4. Using the Task Queue System (Alternative)

For running independent jobs (not training parallelism):

**Start queue server (one terminal):**
```bash
cd ~/cluster-work
python3 queue_server.py
```

**Start workers (other terminals):**
```bash
# Worker1
cd ~/cluster-work
./worker_client.py worker1

# Worker2
ssh worker2 "cd ~/cluster-work && ./worker_client.py worker2"
```

**Add tasks (in queue_server terminal):**
```
> add python train_model.py --config config1.yaml
> add python train_model.py --config config2.yaml
> add python evaluate.py --model model.pth
```

### 5. Simple Job Runner (for quick tests)
```bash
# Run a command on both workers
~/cluster-work/bin/cluster-run "hostname"

# Run a Python script on both workers
~/cluster-work/bin/run-job test_parallel.py
```

## Code Patterns for Distributed ML

### Pattern 1: Model Training Across Multiple Nodes
```python
import ray
import torch
import torch.distributed as dist

@ray.remote
class DistributedTrainer:
    def __init__(self, rank, world_size, data_path):
        self.rank = rank
        self.world_size = world_size
        self.data = self.load_data(data_path, rank, world_size)

    def load_data(self, data_path, rank, world_size):
        # Each worker loads its own portion of the data
        full_data = load_dataset(data_path)
        chunk_size = len(full_data) // world_size
        start = rank * chunk_size
        end = start + chunk_size
        return full_data[start:end]

    def train(self, epochs):
        # Train on local data chunk
        model = create_model()
        for epoch in range(epochs):
            model.train(self.data)
        return model.state_dict()

# Initialize
world_size = 2  # Number of workers
trainers = [DistributedTrainer.remote(i, world_size, "/path/to/data") for i in range(world_size)]

# Train in parallel
model_states = ray.get([trainer.train.remote(10) for trainer in trainers])

# Aggregate or average model states on head node
final_model = average_models(model_states)
```

### Pattern 2: Batch Processing Large Datasets
```python
import ray
import pandas as pd

@ray.remote
def process_batch(batch):
    # Process your batch
    result = batch.mean()
    return result

# Load dataset on head node
df = pd.read_csv("/home/laptopslave1/cluster-work/datasets/large_data.csv")

# Split into batches
batch_size = 10000
batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]

# Process in parallel
futures = [process_batch.remote(batch) for batch in batches]
results = ray.get(futures)

# Combine results
final_result = combine_results(results)
```

### Pattern 3: Real-time Monitoring
```python
import ray
import time

@ray.remote
def monitor_cluster():
    """Monitor cluster status"""
    resources = ray.available_resources()
    return resources

# Monitor every 5 seconds
while True:
    resources = ray.get(monitor_cluster.remote())
    print(f"Available CPUs: {resources.get('CPU', 0)}")
    time.sleep(5)
```

## Troubleshooting

### Check Cluster Status
```bash
# Check Ray cluster
ray status

# Check network connectivity
ping 10.10.10.2  # From Worker1
ping 10.10.10.1  # From Worker2

# Check SSH
ssh worker2 "hostname"
```

### Common Issues

1. **Ray connection fails**:
   ```bash
   # Stop all Ray processes
   ray stop
   # Restart head node
   ray start --head --port=6379
   # Restart worker
   ssh worker2 "ray start --address='10.10.10.1:6379'"
   ```

2. **Dataset not found on worker2**:
   - Copy dataset to both workers for faster access
   - Or use Ray's shared object store: `ray.put(dataset)`

3. **Out of memory**:
   - Reduce batch sizes
   - Use ray's object store: `ray.put()` to avoid duplication
   - Monitor memory: `ray status`

## Best Practices

1. **Always activate virtual environment** before running Python code
2. **Use `ray.put()` for large datasets** to share efficiently
3. **Monitor resources** with `ray status`
4. **Save checkpoints** regularly for long training runs
5. **Use `ray.get()` only when results are needed** to avoid blocking
6. **Stop Ray** when not in use: `ray stop`

## Example: Complete Training Workflow

```python
# train_distributed.py
import ray
import torch
import numpy as np
from datetime import datetime

# Initialize cluster
ray.init(address='auto')

@ray.remote(num_gpus=0)
class Worker:
    def __init__(self, worker_id, data_path):
        self.worker_id = worker_id
        self.data = np.load(data_path)
        print(f"Worker {worker_id} initialized with {len(self.data)} samples")

    def train_epoch(self, epoch, batch_size=32):
        # Simulate training
        accuracy = 0.7 + np.random.random() * 0.2
        return f"Worker {self.worker_id} epoch {epoch}: accuracy={accuracy:.3f}"

# Create workers
workers = [Worker.remote(i, f"/home/laptopslave1/cluster-work/datasets/chunk_{i}.npy") for i in range(2)]

# Run 10 epochs
for epoch in range(10):
    futures = [worker.train_epoch.remote(epoch) for worker in workers]
    results = ray.get(futures)
    for result in results:
        print(result)

ray.shutdown()
print("Training complete!")
```

## Notes for Future LLM Assistance

This cluster has:
- **2 nodes** with Ubuntu 22.04
- **Python 3.12.7** virtual environment at `~/cluster-work/venv3.12.7`
- **Ray** installed for distributed computing
- **Network**: 10.10.10.1 (Worker1) and 10.10.10.2 (Worker2)
- **Data location**: `~/cluster-work/datasets/` on Worker1
- **Code location**: `~/cluster-work/repos/` on Worker1

To run distributed code:
1. SSH to Worker1
2. Activate venv
3. Start Ray if needed
4. Run Python script with ray.init(address='auto')

The cluster is production-ready for ML/DL workloads.
