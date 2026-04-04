# MTGNN
This is forked and adapted for our studies from [temporal_gnn/MTGNN](https://github.com/seferlab/temporal_gnn/tree/master/MTGNN).
The researchers before us forked and adapted it for there studies from [MTGNN](https://github.com/nnzhan/MTGNN.git).

## Data 
The data is already prepared and stored in the data folder, in the [test.csv](data/test/test.csv). This is used for running the [train_fast_single_step_for_speed.py](train_fast_single_step_for_speed.py) and [predict_fast_single_step_for_speed.py](predict_fast_single_step_for_speed.py) scripts. The data is in the same format as the original datasets used and provided by the researchers of the temporal_gnn/MTGNN repository.


## Additional Files Introduced by the temporal_gnn/MTGNN Researchers
- [hyperparameter_finder.py](hyperparameter_finder.py)
- [predict_fast_single_step_for_speed.py](predict_fast_single_step_for_speed.py)
- [train_and_predict_weeks.py](train_and_predict_weeks.py)
- [train_and_predict_weeks_fast.py](train_and_predict_weeks_fast.py)
- [train_fast_single_step_for_speed.py](train_fast_single_step_for_speed.py)
- [predict_weeks.py](predict_weeks.py)
- [run.py](run.py)
- train_weeks.py (This file was not provided but referenced in the [run.py](run.py) file, so we created it with the same code structure as [predict_weeks.py](predict_weeks.py) but without the prediction part and using a similar function available to us from the main class)
- [train_weeks_ibVersion.py](train_weeks_ibVersion.py) (File added by [ib-hussain](https://github.com/ib-hussain) to make the [run.py](run.py) file work for the data provided, for further info refer to the python files named here for details in comments and docstrings)

## System Information & Agent Enhancements (Added by AI Agent)
Through automated documentation efforts, comprehensive docstrings and comments have been added to the codebase within `assignment2work/MTGNN/`. Documentation elements inserted across these scripts feature explicit markings `(Added by AI Agent)`. 

### Affected Files
* **[constants.py](constants.py)**: Module variables like prediction timeframe variables setting fixed baselines.
* **[hyperparameter_finder.py](hyperparameter_finder.py)**: An objective optimizer utilizing Hyperopt specifically running parameters searches optimizing `epoch`, `learning rate`, and structural `channels`.
* **[net.py](net.py)**: Fully encapsulated model logic defining the primary multivariate MTGNN architecture processing spatio-temporal graphs.
* **[predict_fast_single_step_for_speed_testing.py](predict_fast_single_step_for_speed_testing.py)**: Benchmarks output generation running singular prediction evaluation intervals ensuring runtime bounds.
* **[predict_weeks.py](predict_weeks.py)**: Handles long-running prediction segments over defined target weeks sequentially.
* **[run.py](run.py)**: Consolidates runtime sequencing by bridging training updates directly alongside predictive passes logically.
* **[train_and_predict_weeks.py](train_and_predict_weeks.py)**: Unifies dataset iterations for both training structures and concurrent inference pipelines.
* **[train_and_predict_weeks_fast.py](train_and_predict_weeks_fast.py)**: Same iteration objective as above, optimizing skip-patterns enabling significantly sped evaluation cycles.
* **[train_fast_single_step_for_speed_testing.py](train_fast_single_step_for_speed_testing.py)**: Confirms fast operational baseline timings by running basic steps targeting limited horizons minimal impact testing environments.
* **[train_multi_step.py](train_multi_step.py)**: Expands core processing enabling graph optimizations modeling outputs across varied forecast windows natively.
* **[train_single_step.py](train_single_step.py)**: Core layout containing `SingleStep` controller managing graph loading, scaling parameters, modeling evaluation scopes natively.
* **[train_weeks_ibVersion.py](train_weeks_ibVersion.py)**: Configurable weekly handler implementing looping logic bridging data offsets seamlessly over extended datasets natively.
* **[trainer.py](trainer.py)**: Standard logic wrapper managing parameter scaling algorithms alongside standard masked operations dictating `Optim` configurations natively.
* **[util.py](util.py)**: Encompasses fundamental metric derivations (`Dataloader`, `rmse`, `mape`, normalization, standardizations and laplacian routines).
* **[generate_training_data.py](generate_training_data.py)**: Parses original inputs returning validated numpy splits supporting sequence modeling datasets natively.

## side notes
1. If you are getting any error related to the code below, you can comment it out and the code will run absolutely fine, it is just for optimizing the performance of the code and suppressing some warnings that are occuring on my machine.
    ```
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
    ```