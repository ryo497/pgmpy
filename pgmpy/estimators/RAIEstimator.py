#!/usr/bin/env python
from collections import deque
from itertools import permutations
from easydict import EasyDict

import networkx as nx
from tqdm.auto import trange

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.base import PDAG
from pgmpy.base import UndirectedGraph as Graph
from pgmpy.base.visualize_graph import display_graph_info as show
from pgmpy.estimators import (
    AICScore,
    BDeuScore,
    BDsScore,
    BicScore,
    K2Score,
    ScoreCache,
    StructureEstimator,
    StructureScore,
)
from pgmpy.estimators.CITests import NatoriScore
from collections import defaultdict
from itertools import combinations


class RAIEstimator(StructureEstimator):
    """
    Class for heuristic RAI for DAGs, to learn
    network structure from data. `estimate` attempts to find a model with optimal score.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.NaN`.
        Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    use_caching: boolean
        If True, uses caching of score for faster computation.
        Note: Caching only works for scoring methods which are decomposable. Can
        give wrong results in case of custom scoring methods.

    References
    ----------
    Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.4.3 (page 811ff)
    """

    def __init__(self, data, use_cache=True, **kwargs):
        self.use_cache = use_cache
        super(RAIEstimator, self).__init__(data, **kwargs)
    
    @staticmethod
    def skeleton2pdag(skeleton, separating_sets):
        """Orients the edges of a graph skeleton based on information from
        `separating_sets` to form a DAG pattern (DAG).

        Parameters
        ----------
        skeleton: UndirectedGraph
            An undirected graph skeleton as e.g. produced by the
            estimate_skeleton method.

        separating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes then
            conditionally independent. (needed for edge orientation)

        Returns
        -------
        Model after edge orientation: pgmpy.base.DAG
            An estimate for the DAG pattern of the BN underlying the data. The
            graph might contain some nodes with both-way edges (X->Y and Y->X).
            Any completion by (removing one of the both-way edges for each such
            pair) results in a I-equivalent Bayesian network DAG.

        References
        ----------
        Neapolitan, Learning Bayesian Networks, Section 10.1.2, Algorithm 10.2 (page 550)
        http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf


        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import PC
        >>> data = pd.DataFrame(np.random.randint(0, 4, size=(5000, 3)), columns=list('ABD'))
        >>> data['C'] = data['A'] - data['B']
        >>> data['D'] += data['A']
        >>> c = PC(data)
        >>> pdag = c.skeleton_to_pdag(*c.build_skeleton())
        >>> pdag.edges() # edges: A->C, B->C, A--D (not directed)
        [('B', 'C'), ('A', 'C'), ('A', 'D'), ('D', 'A')]
        """

        pdag = skeleton.to_directed()
        node_pairs = list(permutations(pdag.nodes(), 2))

        # 1) for each X-Z-Y, if Z not in the separating set of X,Y, then orient edges as X->Z<-Y
        # (Algorithm 3.4 in Koller & Friedman PGM, page 86)
        for pair in node_pairs:
            X, Y = pair
            if not skeleton.has_edge(X, Y):
                for Z in set(skeleton.neighbors(X)) & set(skeleton.neighbors(Y)):
                    if frozenset((X,Y)) not in separating_sets:
                        pass
                    else:
                        if Z not in separating_sets[frozenset((X, Y))]:
                            pdag.remove_edges_from([(Z, X), (Z, Y)])

        progress = True
        while progress:  # as long as edges can be oriented (removed)
            num_edges = pdag.number_of_edges()

            # 2) for each X->Z-Y, orient edges to Z->Y
            # (Explanation in Koller & Friedman PGM, page 88)
            for pair in node_pairs:
                X, Y = pair
                if not pdag.has_edge(X, Y):
                    for Z in (set(pdag.successors(X)) - set(pdag.predecessors(X))) & (
                        set(pdag.successors(Y)) & set(pdag.predecessors(Y))
                    ):
                        pdag.remove_edge(Y, Z)

            # 3) for each X-Y with a directed path from X to Y, orient edges to X->Y
            for pair in node_pairs:
                X, Y = pair
                if pdag.has_edge(Y, X) and pdag.has_edge(X, Y):
                    for path in nx.all_simple_paths(pdag, X, Y):
                        is_directed = True
                        for src, dst in list(zip(path, path[1:])):
                            if pdag.has_edge(dst, src):
                                is_directed = False
                        if is_directed:
                            pdag.remove_edge(Y, X)
                            break

            # 4) for each X-Z-Y with X->W, Y->W, and Z-W, orient edges to Z->W
            for pair in node_pairs:
                X, Y = pair
                for Z in (
                    set(pdag.successors(X))
                    & set(pdag.predecessors(X))
                    & set(pdag.successors(Y))
                    & set(pdag.predecessors(Y))
                ):
                    for W in (
                        (set(pdag.successors(X)) - set(pdag.predecessors(X)))
                        & (set(pdag.successors(Y)) - set(pdag.predecessors(Y)))
                        & (set(pdag.successors(Z)) & set(pdag.predecessors(Z)))
                    ):
                        pdag.remove_edge(W, Z)

            progress = num_edges > pdag.number_of_edges()

        # TODO: This is temp fix to get a PDAG object.
        edges = set(pdag.edges())
        undirected_edges = []
        directed_edges = []
        for u, v in edges:
            if (v, u) in edges:
                undirected_edges.append((u, v))
            else:
                directed_edges.append((u, v))
        return PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)


    def _legal_operations(
        self,
        model,
        score,
        structure_score,
        tabu_list,
        max_indegree,
        black_list,
        white_list,
        fixed_edges,
    ):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Friedman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `black_list` or `white_list` to exclude those
        edges or to limit the search.
        """

        tabu_list = set(tabu_list)

        # Step 1: Get all legal operations for adding edges.
        potential_new_edges = (
            set(permutations(self.variables, 2))
            - set(model.edges())
            - set([(Y, X) for (X, Y) in model.edges()])
        )

        for X, Y in potential_new_edges:
            # Check if adding (X, Y) will create a cycle.
            if not nx.has_path(model, Y, X):
                operation = ("+", (X, Y))
                if (
                    (operation not in tabu_list)
                    and ((X, Y) not in black_list)
                    and ((X, Y) in white_list)
                ):
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if len(new_parents) <= max_indegree:
                        score_delta = score(Y, new_parents) - score(Y, old_parents)
                        score_delta += structure_score("+")
                        yield (operation, score_delta)

        # Step 2: Get all legal operations for removing edges
        for X, Y in model.edges():
            operation = ("-", (X, Y))
            if (operation not in tabu_list) and ((X, Y) not in fixed_edges):
                old_parents = model.get_parents(Y)
                new_parents = [var for var in old_parents if var != X]
                score_delta = score(Y, new_parents) - score(Y, old_parents)
                score_delta += structure_score("-")
                yield (operation, score_delta)

        # Step 3: Get all legal operations for flipping edges
        for X, Y in model.edges():
            # Check if flipping creates any cycles
            if not any(
                map(lambda path: len(path) > 2, nx.all_simple_paths(model, X, Y))
            ):
                operation = ("flip", (X, Y))
                if (
                    ((operation not in tabu_list) and ("flip", (Y, X)) not in tabu_list)
                    and ((X, Y) not in fixed_edges)
                    and ((Y, X) not in black_list)
                    and ((Y, X) in white_list)
                ):
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = [var for var in old_Y_parents if var != X]
                    if len(new_X_parents) <= max_indegree:
                        score_delta = (
                            score(X, new_X_parents)
                            + score(Y, new_Y_parents)
                            - score(X, old_X_parents)
                            - score(Y, old_Y_parents)
                        )
                        score_delta += structure_score("flip")
                        yield (operation, score_delta)


    
    def RecrusiveSearch(self, Nz, Gs, Gex, Gall, Go, ci_test):
        # print(Nz)
        # Step 1: Initial checks and setup for arguments
        cls_test = ci_test(self.data)
        if all(len(Gs[node]) <= Nz for node in Gs):
            Go.add_nodes_from(Gs.nodes(data=True))
            for node in Gs.nodes:
                Go.add_edges_from(Gall.edges(node, data=True))
            return Go, Gs

        # Step 2: Define the structure_score function
        for node_y in Gs.nodes:
            parents_in_gs = set(Gs.neighbors(node_y))
            for node_x in Gex.node2parents[node_y]:
                Z = parents_in_gs & Gex.node2parents[node_y] - {node_x}
                if len(Z) >= Nz:
                    for Z in combinations(Z, Nz):
                        if cls_test.separate(node_x, node_y, Z):
                            if Gall.has_edge(node_x, node_y):
                                Gall.remove_edge(node_x, node_y)
        # show(Gs)
        if Gex.nodes:
            Gs = self.skeleton2pdag(Gs, cls_test.separating_sets)
        # show(Gall)
        # show(Gs)
        edge_list = []
        for node_y in Gs.nodes:
            neighbors = list(Gs.neighbors(node_y))
            for node_x in neighbors:
                if Nz == 0:
                    Z = []
                    if cls_test.separate(node_x,
                                        node_y,
                                        Z,
                                        self.data,
                                        boolean=True):
                        print("#########")
                        # if Gall.has_edge(node_x, node_y):
                        #     Gall.remove_edge(node_x, node_y)
                        # if Gs.has_edge(node_x, node_y):
                        #     Gs.remove_edge(node_x, node_y)
                else:
                    set_Pa = Gex.node2parents[node_y] | set(neighbors) - {node_x}
                    num_Pa = len(set_Pa)
                    if num_Pa >= Nz:
                        for Z in combinations(set_Pa, Nz):
                            # print(Z)
                            if cls_test.separate(node_x, node_y, Z, self.data, boolean=True):
                                # print(f"独立である {node_x, node_y}")
                                if Gs.has_edge(node_x, node_y):
                                    Gs.remove_edge(node_x, node_y)
                                if Gall.has_edge(node_x, node_y):
                                    Gall.remove_edge(node_x, node_y)

                            # if not cls_test.separate(node_x, node_y, Z, self.data, boolean=True):
                            #     print(f"独立でない {node_x, node_y}")
                            #     edge_list.append((node_x, node_y))
                                # print(edge_list)
                                # break
                        #         if not Gs.has_edge(node_x, node_y):
                        #             Gs.add_edge(node_x, node_y)
                        #         if not Gall.has_edge(node_x, node_y):
                        #             Gall.add_edge(node_x, node_y)
                        #         break
                        # if Gs.has_edge(node_x, node_y):
                        #     Gs.remove_edge(node_x, node_y)
                        # if Gall.has_edge(node_x, node_y):
                        #     Gall.remove_edge(node_x, node_y)
                        # if Gall.has_edge(node_x, node_y):
                        #     Gall.remove_edge(node_x, node_y)
                        # if Gs.has_edge(node_x, node_y):
                        #     Gs.remove_edge(node_x, node_y)
                                # if Gs.has_edge(node_x, node_y):
                                #     Gs.remove_edge(node_x, node_y)
                                # if Gall.has_edge(node_x, node_y):
                                #     Gall.remove_edge(node_x, node_y)
                                # if Gall.has_edge(node_x, node_y):
                                #     Gall.remove_edge(node_x, node_y)
                                # if Gs.has_edge(node_x, node_y):
                                #     Gs.remove_edge(node_x, node_y)
        # この項大事そう…
        # print(edge_list)
        nodes = Gs.nodes
        # Gs = Graph()
        # Gs.add_nodes_from(nodes)
        # Gs.add_edges_from(edge_list)
        # Gall = Gs.copy()
        # show(Gs)
        Gs = self.skeleton2pdag(Gs, cls_test.separating_sets)
        Gall = self.skeleton2pdag(Gall, cls_test.separating_sets)
        # show(Gs)
        independent_nodes = nodes - Gs.nodes
        if independent_nodes:
            Go.add_nodes_from(independent_nodes)
        Gd, g_subs, Gex = self.order_grouping(Gs, Gex)
        for subs in g_subs:
            Go, gs = self.RecrusiveSearch(Nz + 1, subs, Gex, Gall, Go, ci_test)
        return self.RecrusiveSearch(Nz + 1, Gd, Gex, Gall, Go, ci_test)


    def order_grouping(self, Gs, Gex):
        sccs = list(nx.strongly_connected_components(Gs))
        scc_graph = nx.condensation(Gs)
        # print(Gs)
        topological_order = list(nx.topological_sort(scc_graph))
        # print(topological_order)
        topological_lowest_nodes = sccs[topological_order[0]]
        gc = self.extract_subgraph(
            topological_lowest_nodes,
            Gs
        )
        Gex.nodes |= Gs.nodes - set(topological_lowest_nodes)
        SubStructures = []
        for sccs_idx in topological_order[1:]:
            sub = self.extract_subgraph(
                sccs[sccs_idx],
                Gs
            )
            SubStructures.append(sub)
        
        Gex = self.update_gex(
            Gex,
            sccs,
            scc_graph,
            SubStructures,
            topological_order
        )
        return gc, SubStructures, Gex

    def update_gex(
            self,
            gex,
            sccs,
            scc_graph,
            substructures,
            topological_order
    ):
        current_sets = set.union(*sccs)
        for idx, scc in enumerate(sccs):
            for node in scc:
                parent_sets = gex.node2parents[node]
                fixed_sets = parent_sets - current_sets
                update_sets = set()
                for parent_idx in scc_graph[idx]:
                    update_sets |= sccs[parent_idx]
                gex.node2parents[node] = fixed_sets | update_sets
        return gex


    def extract_subgraph(self, nodes, G):
        sub = Graph()
        sub.add_nodes_from(nodes)
        edges_to_add = [(node, sub_node) for node in nodes for sub_node in nodes if node != sub_node and G.has_edge(node, sub_node)]
        sub.add_edges_from(edges_to_add)
        return sub

    def estimate(
        self,
        scoring_method="natoriscore",
        Gs=None,
        fixed_edges=set(),
        max_indegree=None,
        show_progress=True,
    ):

        # Step 1: Initial checks and setup for arguments
        # Step 1.1: Check scoring_method
        supported_methods = {
            "k2score": K2Score,
            "bdeuscore": BDeuScore,
            "bdsscore": BDsScore,
            "bicscore": BicScore,
            "aicscore": AICScore,
            "natoriscore": NatoriScore,
        }
        if (
            (
                isinstance(scoring_method, str)
                and (scoring_method.lower() not in supported_methods)
            )
        ) and (not isinstance(scoring_method, StructureScore)):
            raise ValueError(
                "scoring_method should either be one of k2score, bdeuscore, bicscore, bdsscore, aicscore, or an instance of StructureScore"
            )

        if isinstance(scoring_method, str):
            ci_test = supported_methods[scoring_method.lower()]
        else:
            ci_test = scoring_method

        # if self.use_cache:
        #     score_fn = ScoreCache.ScoreCache(score, self.data).local_score
        # else:
        #     score_fn = score.local_score

        # Step 1.2: Check the start_dag
        if Gs is None:
            Gs = Graph()
            Gs.add_nodes_from(self.variables)
            Gs.add_edges_from(
                [
                    (X, Y)
                    for X, Y in combinations(self.variables, 2)
                    if X != Y
                ]
            )
            Gs.complete_graph()
        elif not isinstance(Gs, Graph):
            raise ValueError(
                "'start_dag' should be a skeleton with the same variables as the data set, or 'None'."
            )

        # Step 1.3: Check fixed_edges
        if not hasattr(fixed_edges, "__iter__"):
            raise ValueError("fixed_edges must be an iterable")
        else:
            fixed_edges = set(fixed_edges)
            # Gs.add_edges_from(fixed_edges)
            # if not nx.is_directed_acyclic_graph(start_dag):
            #     raise ValueError(
            #         "fixed_edges creates a cycle in start_dag. Please modify either fixed_edges or start_dag."
            #     )
        current_model = Gs
        # print(Gs.__dict__.keys())
        # print(Gs._succ)
        # hoge
        Gall = Gs.copy()
        Go = DAG()
        gex = EasyDict()
        gex.nodes = set()
        gex.node2parents = {var: set() for var in self.variables}
        Nz = 0

        # Step 2: Define the structure_score function
        best_model = self.RecrusiveSearch(
            Nz=Nz,
            Gs=current_model,
            Gex=gex,
            Go=Go,
            Gall=Gall,
            ci_test=ci_test
        )

        return best_model
    
    def initial_model(self):
        Gs = Graph()
        Gs.add_nodes_from(self.variables)
        Gs.add_edges_from(
            [
                (X, Y)
                for X, Y in combinations(self.variables, 2)
                if X != Y
            ]
        )
        Gs.complete_graph()
        return Gs

