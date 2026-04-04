'''
ib-hussain: This file needs to be run from the CLI and it has default args so they are configured and the paths are also configured.

(Added by AI Agent)
Module: train_fast_single_step_for_speed_testing.py
Automates a performance benchmarking run testing the SingleStep's training efficiency.
Captures performance metrics across reduced batch settings specifically scaled for fast analysis.
'''
from train_single_step import SingleStep, print_results

import os
import dotenv
dotenv.load_dotenv()
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations for CPU
# Enable XLA compilation for faster CPU execution
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
# Set CPU affinity for better performance
try:
    tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count() - 1)  # Leave 1 core for system
    tf.config.threading.set_inter_op_parallelism_threads(2)  # For parallel ops
except:
    pass

def main(device):
    """
    Initiates multiple sparse training runs capturing cumulative operational metrics.
    (Added by AI Agent)
    """
    single_step = get_single_step(device)
    vacc, vrae, vcorr, vmape, acc, rae, corr, mape = [], [], [], [], [], [], [], []
    all_metrics_arrays = [vacc, vrae, vcorr, vmape, acc, rae, corr, mape]
    runs = 2
    for i in range(runs):
        all_metrics = single_step.run()
        [a.append(m) for a, m in zip(all_metrics_arrays, all_metrics)]
    print_results(runs, acc, corr, mape, rae, vacc, vcorr, vmape, vrae)
def get_single_step(device):
    """
    Generates a SingleStep iteration configured for minimally complex test executions.
    (Added by AI Agent)
    """
    return SingleStep(
        data_path=f"{str(os.getenv('datasets_MTGNN_path', "assignment2work/MTGNN/data"))}/test/test.csv",
        week=3,
        num_weeks=3,
        device=device,
        num_nodes=20,
        subgraph_size=3,
        seq_in_len=5,
        horizon=5,
        batch_size=30,
        epochs=2,
    )
if __name__ == "__main__":
    main(f"{str(os.getenv('PROCESSOR', 'cpu'))}")
