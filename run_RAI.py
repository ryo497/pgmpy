import pandas as pd
import numpy as np
import networkx as nx
import argparse
import os
from pgmpy.base.visualize_graph import display_graph_info
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators.RAIEstimator import RAIEstimator
from pgmpy.estimators import BicScore, HillClimbSearch
from pgmpy.estimators import PC
from pgmpy.estimators.CITests import *
import time
SAVE_DIR = "./results_2"

ESTIMATOR={
    "RAI": RAIEstimator,
    "HC": HillClimbSearch,
    "PC": PC
}
SCORE={
    "BIC": BicScore
}

def load_data(type, sample_size):
    model = get_example_model(type)
    sampler = BayesianModelSampling(model)
    data = sampler.forward_sample(size=sample_size)
    # print(data.head())
    return model, data

def test_benchmark(
        estimate_type,
        data_type,
        sample_size,
        structure_score,
        max_iter,
    ):
    calc_time = 0
    ave_score = 0
    for i in range(max_iter):
        _ , data = load_data(data_type, sample_size)
        estimator = ESTIMATOR[estimate_type](data)
        score = SCORE[structure_score](data)
        t = time.time()
        if estimate_type == "RAI":
            best_model, _ = estimator.estimate()
            calc_time += time.time() - t
            iter_score = score.score(best_model)
            print(f"iteration {i}: {structure_score}:{iter_score}")
            ave_score += iter_score
        elif estimate_type == "HC":
            hc_model = estimator.estimate(scoring_method=score)
            calc_time += time.time() - t
            iter_score = score.score(hc_model)
            print(f"iteration {i}: {structure_score}:{iter_score}")
            ave_score += iter_score
        elif estimate_type == "PC":
            pc_model = estimator.estimate()
            calc_time += time.time() - t
            iter_score = score.score(pc_model)
            print(f"iteration {i}: {structure_score}:{iter_score}")
            ave_score += iter_score
        else:
            raise ValueError("Invalid estimator type")
    ave_score /= max_iter
    calc_time /= max_iter
    save_benchmark(estimate_type, data_type, sample_size, structure_score, ave_score, calc_time)


def save_benchmark(estimate_type, data_type, size, structure_score, ave_score, calc_time):
    results = pd.DataFrame(columns=["estimate_type", "data_type", "structure_score", "ave_score", "calc_time"])
    results.loc[0] = [estimate_type, data_type, structure_score, ave_score, calc_time]
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    file_path = f"{SAVE_DIR}/{estimate_type}_{data_type}_{size}_{structure_score}"
    file_exists = os.path.isfile(file_path)
    with open(file_path, "w") as f:
        results.to_csv(f, header=not file_exists, index=False)


def arg_parser():
    parser = argparse.ArgumentParser(description="Benchmarking for Bayesian Network Structure Learning")
    parser.add_argument("--estimate_type", type=str, default="RAI", help="Estimator type")
    parser.add_argument("--data_type", type=str, default="asia", help="Data type")
    parser.add_argument("--sample_size", type=int, default=10000, help="Sample size")
    parser.add_argument("--structure_score", type=str, default="BIC", help="Structure score")
    parser.add_argument("--max_iter", type=int, default=10, help="Number of iterations")
    return parser.parse_args()

def main():
    args = arg_parser()
    data_type = args.data_type
    sample_size = args.sample_size
    max_iter = args.max_iter
    estimate_type = args.estimate_type
    structure_score = args.structure_score
    file_path = f"{SAVE_DIR}/{estimate_type}_{data_type}_{sample_size}_{structure_score}"
    file_exists = os.path.isfile(file_path)
    # if file_exists:
    #     return
    test_benchmark(estimate_type,
                   data_type,
                   sample_size,
                   structure_score,
                   max_iter)
    # data_type = "sachs"
    # sample_size = 100000
    # model, data = load_data(data_type, sample_size)
    # t = time.time()
    # estimator = RAIEstimator(data)
    # best_model, _ = estimator.estimate()
    # calc_time = time.time() - t
    # display_graph_info(best_model)
    # bic = BicScore(data)

    # # RAIモデルのBICスコアを計算
    # # rai_bic_score = bic.score(model)
    # rai_bic_score = bic.score(best_model)
    # print("BIC Score for the RAI model:", rai_bic_score)
    # print("Calculation Time:", calc_time)
    # display_graph_info(best_model)
    # hc = HillClimbSearch(data)
    # hc_model = hc.estimate(scoring_method=BicScore(data))

    # # BICスコアの計算
    # hc_bic_score = bic.score(hc_model)
    # print("BIC Score for the HC model:", hc_bic_score)

    # pc = PC(data)
    # pc_model = pc.estimate()

    # # PCモデルのBICスコアを計算
    # pc_bic_score = bic.score(pc_model)
    # print("BIC Score for the PC model:", pc_bic_score)



if __name__ == "__main__":
    main()
