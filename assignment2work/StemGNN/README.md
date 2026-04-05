
## Additional Files Introduced by the temporal_gnn/MTGNN Researchers
- [crypto_daily_marked.csv](dataset/crypto_daily_marked.csv)
- [fx_daily_marked.csv](dataset/fx_daily_marked.csv)
- [dataset_preparer.py](dataset_preparer.py)
- [pipeline.py](pipeline.py)
- [runner.py](runner.py)
- [hyperparameter_finder.py](hyperparameter_finder.py)
- [my_evaluator.py](my_evaluator.py)
- [output_formatter_for_portfolio_analysis.py](output_formatter_for_portfolio_analysis.py)

## Order of running the code
- [dataset_preparer.py](dataset_preparer.py)
- [pipeline.py](pipeline.py)
- [runner.py](runner.py)
- [main.py](main.py)
- [my_evaluator.py](my_evaluator.py)
- [output_formatter_for_portfolio_analysis.py](output_formatter_for_portfolio_analysis.py)
- [hyperparameter_finder.py](hyperparameter_finder.py)

## Training and Evaluation

The training procedure and evaluation procedure are all included in the `main.py`. To train and evaluate on some dataset, run the following command:

```train & evaluate
python main.py --train True --evaluate True --dataset <name of csv file> --output_dir <path to output directory> --n_route <number of nodes> --window_size <length of sliding window> --horizon <predict horizon> --norm_method z_score --train_length 7 --validate_length 2 --test_length 1
```

The detailed descriptions about the parameters are as following:

| Parameter name | Description of parameter |
| --- | --- |
| train | whether to enable training, default True |
| evaluate | whether to enable evaluation, default True |
| dataset | file name of input csv |
| window_size | length of sliding window, default 12 |
| horizon | predict horizon, default 3 |
| train_length | length of training data, default 7 |
| validate_length | length of validation data, default 2 |
| test_length | length of testing data, default 1 |
| epoch | epoch size during training |
| lr | learning rate |
| multi_layer | hyper parameter of STemGNN which controls the parameter number of hidden layers, default 5 |
| device | device that the code works on, 'cpu' or 'cuda:x' | 
| validate_freq | frequency of validation |
| batch_size | batch size |
| norm_method | method for normalization, 'z_score' or 'min_max' |
| early_stop | whether to enable early stop, default False |

