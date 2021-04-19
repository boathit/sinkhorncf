
### Code organization

* `julia/costMatrix.jl` generates the items cost matrix from the training datasets. 
* `utils/util.py` holds the utility functions like read, split, and performance measures (ndcg, recall).
* `utils/movideLens.py` and `utils/netflix.py` are two preprocessing scripts.
* `tch/sinkhorn.py` implements the accelerated Sinkhorn loss in a numerical stable way.
* `tch/model.py` implements the models used in `train.py` and `evaluation.py`
* `tch/train.py` trains and saves the best model in terms of `ndcg@100` on validation datasets.
* `tch/evaluation.py` evaluates the best model performance on testing datasets.

### Prerequisite

* Ubuntu OS 18.04
* Julia 1.5
* Python 3.8 + PyTorch 1.5

* Run command like `julia -e 'using Pkg; Pkg.add("CSV")'` to add all required packages which can be found in `julia/costMatrix.jl`.

### Datasets and preprocessing

* [MovieLens 20m](https://grouplens.org/datasets/movielens/20m/)
* [Netflix](https://www.kaggle.com/netflix-inc/netflix-prize-data)

Once the datasets are downloaded, put them into an appropriate locations and rename the unzipped folder names, e.g., the code uses "/home/xx/data/ml-20m".

You are supposed to replace xx with your username in the path `/home/xx/data/...` in `julia/costMatrix.jl`, `utils/movieLens.py`, `utils/netflix.py`, `tch/train.py`, `tch/evaluation.py`.


* Run the script `python -m utils.movieLens` to get the input data for the model.

This will generate files `unique_sid.txt`, `train.csv`, `validation_tr.csv`, `validation_te.csv`, `test_tr.csv`, `test_te.csv`.

* `unique_sid.txt` is used for helping to read `train.csv` and it is optional.
* `train.csv` contains the training data.
* `validation_tr.csv` is used as input for the model in validation stage.
* `validation_te.csv` is used for testing the model output in validation stage.
* `test_tr.csv`, `test_te.csv` are the same as `validation_tr.csv`, `validation_te.csv` but used in testing stage.

* Run `julia julia/costMatrix.jl --dataset ml-20m --datasize large` to produce item cost matrix.

### Train and evaluation

* Run `python -m tch.train -dataset ml-20m` to train the model.

* Run `python -m tch.evaluation -dataset ml-20m` to evaluate the model using the best model obtained in the training.

The best model obtained in training will be saved by default at `/tmp/torch/sinkhorncf/{dataset}/large/model.pt`.