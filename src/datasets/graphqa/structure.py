"""Graph structure recovered from GraphQA prompt text.

Two consumers:
  * the dataset generator, which writes WL labels into the prompt (token-space
    arm, see ``graph_text_encoders.adjacency_wl_encoder``);
  * the collator, which tags every node-mention token with a node row and a WL
    row (embedding-space arm, see ``src.models.hrm_text.HRMGraphICL``).

Parsing assumes the ``adjacency`` encoder with integer node names. A prompt may
hold several graphs (meta-icl few-shot examples); each "In an (un)directed
graph" header opens a new segment whose node ids are namespaced to it.
Only three places count as node mentions: the node list, the first two fields
of each edge tuple (a third field is a capacity), and the ``Q:`` line.
"""

import re
import zlib

import networkx as nx
import numpy as np

GRAPH_HEADER = re.compile(r"In an? (?:un)?directed graph")
NODE_LIST = re.compile(r"among nodes ([^\n]*?)\.\n")
EDGE_TUPLE = re.compile(r"\((\d+), (\d+)(?:, (\d+))?\)")
Q_LINE = re.compile(r"Q: [^\n]*")
INTEGER = re.compile(r"\d+")


def wl_labels(nnodes: int, edges) -> list[int]:
  """Ordered 1-WL colour refinement run to convergence.

  Every node starts with label 0. Each round, a node's signature is its current
  label followed by the sorted multiset of its neighbours' labels. New labels
  are the ranks of the signatures in sorted order. Keying on the old label
  first means each round refines the previous partition. Labels are canonical,
  meaning the same for isomorphic inputs, and after round 1 they rank-order degree.
  Refinement stops when the number of classes no longer grows.
  """
  graph = nx.Graph()
  graph.add_nodes_from(range(nnodes))
  graph.add_edges_from(edges)
  labels = [0] * nnodes
  n_classes = 1
  for _ in range(nnodes):
    sigs = [
        (labels[v], tuple(sorted(labels[u] for u in graph.neighbors(v))))
        for v in range(nnodes)
    ]
    rank = {sig: i for i, sig in enumerate(sorted(set(sigs)))}
    labels = [rank[s] for s in sigs]
    if len(rank) == n_classes:
      break
    n_classes = len(rank)
  return labels


def parse_graph_segments(text: str) -> list[dict]:
  """Split ``text`` into graph segments and locate node mentions in each.

  Returns one dict per segment with keys ``nnodes``, ``edges`` (list of
  ``(u, v)``), and ``mentions`` (list of ``(char_start, char_end, node)``).
  """
  starts = [m.start() for m in GRAPH_HEADER.finditer(text)]
  segments = []
  for k, start in enumerate(starts):
    end = starts[k + 1] if k + 1 < len(starts) else len(text)
    node_list = NODE_LIST.search(text, start, end)
    if node_list is None:
      continue
    mentions = [
        (node_list.start(1) + m.start(), node_list.start(1) + m.end(), int(m.group()))
        for m in INTEGER.finditer(node_list.group(1))
    ]
    nnodes = len(mentions)
    edges = []
    for m in EDGE_TUPLE.finditer(text, node_list.end(), end):
      edges.append((int(m.group(1)), int(m.group(2))))
      mentions.append((m.start(1), m.end(1), int(m.group(1))))
      mentions.append((m.start(2), m.end(2), int(m.group(2))))
    q_line = Q_LINE.search(text, node_list.end(), end)
    if q_line is not None:
      mentions.extend(
          (q_line.start() + m.start(), q_line.start() + m.end(), int(m.group()))
          for m in INTEGER.finditer(q_line.group())
      )
    bad = [node for _, _, node in mentions if node >= nnodes]
    if bad:
      raise ValueError(f"node ids {bad} exceed the {nnodes}-node list")
    segments.append({"nnodes": nnodes, "edges": edges, "mentions": mentions})
  return segments


class GraphStructureFeaturizer:
  """Maps a prompt to ``(char_start, char_end, node_row, wl_row)`` spans.

  node_mode:
    ``perm``     each segment draws a random permutation pi over
                 ``num_node_rows`` rows and node v gets row pi(v), so only the
                 equality of rows carries information (co-reference).
    ``shuffled`` every mention draws an independent row. Co-reference is
                 destroyed, but the digits of one mention still share a row
                 (control).
  wl_mode:
    ``true``     row = the node's WL label.
    ``shuffled`` WL labels permuted across the nodes of the segment (control).

  With ``deterministic=True`` the randomness is seeded from ``(seed, prompt)``,
  so evaluation is reproducible. Otherwise each call draws fresh OS entropy.
  That keeps DataLoader workers from sharing a stream and gives the example a
  new pi every time it is seen.
  """

  def __init__(self, num_node_rows=64, num_wl_rows=64, node_mode="perm",
               wl_mode="true", deterministic=False, seed=0):
    if node_mode not in ("perm", "shuffled"):
      raise ValueError(f"Unknown node_mode: {node_mode}")
    if wl_mode not in ("true", "shuffled"):
      raise ValueError(f"Unknown wl_mode: {wl_mode}")
    self.num_node_rows = num_node_rows
    self.num_wl_rows = num_wl_rows
    self.node_mode = node_mode
    self.wl_mode = wl_mode
    self.deterministic = deterministic
    self.seed = seed

  def __call__(self, text: str) -> list[tuple[int, int, int, int]]:
    rng = (np.random.default_rng([self.seed, zlib.crc32(text.encode())])
           if self.deterministic else np.random.default_rng())
    spans = []
    for seg in parse_graph_segments(text):
      nnodes = seg["nnodes"]
      if nnodes > self.num_node_rows:
        raise ValueError(f"{nnodes} nodes > num_node_rows={self.num_node_rows}")
      perm = rng.permutation(self.num_node_rows)[:nnodes]
      wl = wl_labels(nnodes, seg["edges"])
      if self.wl_mode == "shuffled":
        wl = list(rng.permutation(wl))
      for start, end, node in seg["mentions"]:
        if self.node_mode == "perm":
          node_row = int(perm[node])
        else:
          node_row = int(rng.integers(self.num_node_rows))
        spans.append((start, end, node_row, min(int(wl[node]), self.num_wl_rows - 1)))
    return spans
