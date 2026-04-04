'''
ib-hussain: This file needs to be run from CLI and it's paths are configured.
It is used to test the speed of the prediction function for a single step, as opposed to the weekly model which predicts multiple steps at once.

(Added by AI Agent)
Module: predict_fast_single_step_for_speed_testing.py
This script performs a fast, single-step prediction for speed testing and profiling. 
It configures CPU settings and uses the SingleStep class specifically in prediction mode.
'''
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
from train_single_step import SingleStep

def main(device):
    """
    Main function to run the speedy prediction test.
    (Added by AI Agent)

    Args:
        device (str): Device to use for running the prediction (e.g., 'cpu').
    """
    get_single_step(device).predict_with_the_best_model()
def get_single_step(device):
    """
    Helper function to initialize the SingleStep trainer/predictor in prediction mode.
    (Added by AI Agent)

    Args:
        device (str): Device to use for the prediction.
    
    Returns:
        SingleStep: Initialized predictor object specifically configured for speed testing.
    """
    return SingleStep(
        data_path=f"{str(os.getenv('datasets_MTGNN_path', 'assignment2work/MTGNN/data'))}/test/test.csv",
        week=3,
        num_weeks=3,
        device=device,
        num_nodes=20,
        subgraph_size=8,
        seq_in_len=5,
        horizon=7,
        batch_size=30,
        epochs=30,
        run_for_prediction=True,
    )
if __name__ == "__main__":
    main(f"{str(os.getenv('PROCESSOR', 'cpu'))}")
