'''
ib-hussain: This file needs to be run from CLI and it's paths are configured.
It is used to test the speed of the prediction function for a single step, as opposed to the weekly model which predicts multiple steps at once.
'''
import os
import dotenv
dotenv.load_dotenv()
from train_single_step import SingleStep

def main(device):
    get_single_step(device).predict_with_the_best_model()
def get_single_step(device):
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
