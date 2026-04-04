'''
ib-hussain: This file needs to be run from the CLI and it has default args so they are configured and the paths are also configured.
'''
from train_single_step import SingleStep, print_results
import os
import dotenv
dotenv.load_dotenv()

def main(device):
    single_step = get_single_step(device)
    vacc, vrae, vcorr, vmape, acc, rae, corr, mape = [], [], [], [], [], [], [], []
    all_metrics_arrays = [vacc, vrae, vcorr, vmape, acc, rae, corr, mape]
    runs = 2
    for i in range(runs):
        all_metrics = single_step.run()
        [a.append(m) for a, m in zip(all_metrics_arrays, all_metrics)]
    print_results(runs, acc, corr, mape, rae, vacc, vcorr, vmape, vrae)
def get_single_step(device):
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
