import pandas as pd
import numpy as np
import networkx as nx
from pgmpy.base.visualize_graph import display_graph_info
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators.RAIEstimator import RAIEstimator
from pgmpy.estimators import BicScore, HillClimbSearch
from pgmpy.estimators.CITests import *

def load_data(type):
    model = get_example_model(type)
    sampler = BayesianModelSampling(model)
    data = sampler.forward_sample(size=100)
    # print(data.head())
    return model, data


def main():
    data_type = "asia"
    # data_type = "alarm"
    model, data = load_data(data_type)
    estimator = RAIEstimator(data)
    best_model, _ = estimator.estimate()
    display_graph_info(best_model)
    bic = BicScore(data)

    # RAIモデルのBICスコアを計算
    rai_bic_score = bic.score(best_model)
    print("BIC Score for the RAI model:", rai_bic_score)
    # display_graph_info(best_model)
    hc = HillClimbSearch(data)
    hc_model = hc.estimate(scoring_method=BicScore(data))

    # BICスコアの計算
    hc_bic_score = bic.score(hc_model)
    print("BIC Score for the HC model:", hc_bic_score)

    from pgmpy.estimators import PC
    pc = PC(data)
    pc_model = pc.estimate()

    # PCモデルのBICスコアを計算
    pc_bic_score = bic.score(pc_model)
    print("BIC Score for the PC model:", pc_bic_score)



if __name__ == "__main__":
    main()
