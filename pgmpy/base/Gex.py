from typing import Any
import networkx as nx
from networkx import DiGraph
from collections import defaultdict


# class Gex:
#     def __init__(self):
#         self.substructures = set()
#         self.node2parents = defaultdict(set)
#         self.nodes = set()


#     def node(self):
#         return self.nodes


#     def add_Pa_data(self,
#                     sccs,
#                     scc_graph,
#                     substructures,
#                     topological_order) -> Any:
#         for idx, scc in enumerate(sccs):
#             for node in scc:
#                 parent_s = set()
#                 for parent_idx in scc_graph[idx]:
#                     parent_s |= sccs[parent_idx]
#                 self.node2parents[node] |= parent_s
#                 # print(f"Node: {node}, Parents: {parent_s}")
#         if self.nodes == set():
#             for idx in topological_order[1:]:
#                 self.nodes |= set(sccs[idx])
#         self.substructures |= set(substructures)
#         return

#     def parents(self, node):
#         return self.node2parents[node]
