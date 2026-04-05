import argparse
'''
ib-hussain: The default paths are properly set over here.
'''
import pandas as pd
from tqdm import tqdm
import os
import dotenv
dotenv.load_dotenv()
datasets_pathi = str(os.getenv("datasets_StemGNN_path", "assignment2work/StemGNN/datasets"))
ENDING_WEEK = int(os.getenv("ENDING_WEEK", "21"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="crypto_daily_marked")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--num_weeks", type=int, help="Total number of weeks", default=ENDING_WEEK)
    parser.add_argument(
        "--num_weeks_needed",
        help="Needed number of weeks for prediction",
        type=int,
        default=ENDING_WEEK,
    )
    return parser.parse_args()
def main(dataset, horizon, num_weeks, num_weeks_needed):
    dataset_path = f"{datasets_pathi}/{dataset}.csv"
    for week in tqdm(range(1, num_weeks_needed + 1), desc="dataset_preparer.py > week"):
        prices_np = read_raw_data(path=dataset_path, week=week, num_weeks=num_weeks)
        simple_returns_np = _convert_to_simple_returns(prices_np, horizon)
        output_file_name = (f"{datasets_pathi}/__{dataset}_simple_returns_week_{week}.csv".replace("_marked", ""))
        pd.DataFrame(simple_returns_np).to_csv(output_file_name, index=False, header=False)
def read_raw_data(path, week, num_weeks):
    fin = open(path)
    raw_data = pd.read_csv(fin)

    truncate_index = len(raw_data)
    stopping_point = -1
    truncation_mark = num_weeks - week

    for point in reversed(raw_data["split_point"]):
        if stopping_point == truncation_mark:break
        if point:stopping_point += 1
        truncate_index -= 1
    return (raw_data.loc[:truncate_index].drop(["split_point"], axis=1).drop(columns=["Date"]).to_numpy())
def _convert_to_simple_returns(prices_np, horizon):
    simple_returns = prices_np[horizon:] / prices_np[:-horizon]
    return simple_returns
if __name__ == "__main__":
    args = parse_args()
    main(args.dataset, args.horizon, args.num_weeks, args.num_weeks_needed)

# python assignment2work/StemGNN/dataset_preparer.py --dataset crypto_daily_marked > assignment2work/StemGNN/dataset_preparer_crypto_results.txt
# python assignment2work/StemGNN/dataset_preparer.py --dataset fx_daily_marked > assignment2work/StemGNN/dataset_preparer_fx_results.txt