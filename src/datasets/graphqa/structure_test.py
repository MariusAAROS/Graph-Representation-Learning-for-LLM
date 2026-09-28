"""Tests for structure.py and the structure-aware collator.

Run from the repo root: python -m src.datasets.graphqa.structure_test
Data-dependent tests skip when data/graphqa/baseline-struct is not generated,
and tokenizer tests skip when HRM-Text-1B is not in the HF cache.
"""

import json
import os
import random
import re
import unittest

import networkx as nx
import torch

from src.datasets.graphqa import structure
from src.datasets.loaders import BaselineDataset, make_collator

DATA = "data/graphqa/{}/standard/{}.json"
HRM = "sapientinc/HRM-Text-1B"


def _records(config, split):
  path = DATA.format(config, split)
  if not os.path.exists(path):
    raise unittest.SkipTest(f"{path} not generated")
  with open(path) as f:
    return json.load(f)


def _tokenizer():
  try:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(HRM, use_fast=True)
  except OSError as err:
    raise unittest.SkipTest(f"tokenizer unavailable: {err}")


class WLTest(unittest.TestCase):

  def test_path_graph(self):
    # 0-1-2-3-4: endpoints, their neighbours, and the centre are three classes.
    labels = structure.wl_labels(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    self.assertEqual(labels[0], labels[4])
    self.assertEqual(labels[1], labels[3])
    self.assertEqual(len(set(labels)), 3)

  def test_first_round_orders_degree(self):
    graph = nx.star_graph(4)  # centre 0 has degree 4, leaves degree 1
    labels = structure.wl_labels(5, list(graph.edges()))
    self.assertGreater(labels[0], labels[1])

  def test_isomorphism_invariant(self):
    rng = random.Random(0)
    for _ in range(50):
      graph = nx.gnp_random_graph(9, 0.35, seed=rng.randint(0, 10**6))
      perm = list(range(9))
      rng.shuffle(perm)
      relabelled = [(perm[u], perm[v]) for u, v in graph.edges()]
      labels = structure.wl_labels(9, list(graph.edges()))
      labels_perm = structure.wl_labels(9, relabelled)
      self.assertEqual([labels[v] for v in range(9)],
                       [labels_perm[perm[v]] for v in range(9)])


class ParseTest(unittest.TestCase):

  def test_edges_round_trip(self):
    for split in ("train", "val", "test"):
      for r in _records("baseline-struct", split):
        (seg,) = structure.parse_graph_segments(r["question"])
        self.assertEqual(seg["edges"], [tuple(e) for e in r["edges"]], r["id"])
        self.assertEqual(seg["nnodes"], int(r["nnodes"]), r["id"])

  def test_mentions_are_node_names(self):
    for r in _records("baseline-struct", "test"):
      (seg,) = structure.parse_graph_segments(r["question"])
      for start, end, node in seg["mentions"]:
        self.assertEqual(r["question"][start:end], str(node))
      # nnodes (list) + 2 * nedges (tuples) + node ids in the Q line.
      q_line = structure.Q_LINE.search(r["question"]).group()
      n_q = len(re.findall(r"\d+", q_line))
      self.assertEqual(len(seg["mentions"]), int(r["nnodes"]) + 2 * int(r["nedges"]) + n_q)

  def test_max_flow_prints_capacities(self):
    for r in _records("baseline-struct", "test"):
      tuples = structure.EDGE_TUPLE.findall(r["question"])
      if r["task"] == "MaximumFlow" and tuples:
        self.assertTrue(all(c for _, _, c in tuples), r["id"])
      elif r["task"] != "MaximumFlow":
        self.assertFalse(any(c for _, _, c in tuples), r["id"])

  def test_wl_variants_share_graphs(self):
    wl_line = re.compile(r"The structural labels of the nodes are: [^\n]*\n")
    base = _records("baseline-struct", "test")
    for config in ("baseline-wl", "baseline-wlshuf"):
      other = _records(config, "test")
      for a, b in zip(base, other):
        self.assertEqual(a["question"], wl_line.sub("", b["question"]))
        self.assertEqual(a["answer"], b["answer"])

  def test_wl_text_matches_structure(self):
    pair = re.compile(r"(\d+): (\d+)")
    for r in _records("baseline-wl", "test"):
      line = re.search(r"structural labels of the nodes are: ([^\n]*)", r["question"])
      text_labels = [int(l) for _, l in pair.findall(line.group(1))]
      self.assertEqual(text_labels, structure.wl_labels(int(r["nnodes"]), r["edges"]))


class FeaturizerTest(unittest.TestCase):

  def test_perm_is_consistent_and_injective(self):
    featurize = structure.GraphStructureFeaturizer(deterministic=True)
    for r in _records("baseline-struct", "test")[:300]:
      spans = featurize(r["question"])
      row_of = {}
      for start, end, row, _ in spans:
        node = int(r["question"][start:end])
        self.assertEqual(row_of.setdefault(node, row), row)
      self.assertEqual(len(set(row_of.values())), len(row_of))

  def test_deterministic_mode_is_reproducible(self):
    text = _records("baseline-struct", "test")[0]["question"]
    a = structure.GraphStructureFeaturizer(deterministic=True, seed=3)(text)
    b = structure.GraphStructureFeaturizer(deterministic=True, seed=3)(text)
    c = structure.GraphStructureFeaturizer(deterministic=True, seed=4)(text)
    self.assertEqual(a, b)
    self.assertNotEqual(a, c)


class CollatorTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tokenizer = _tokenizer()
    self.batch = [BaselineDataset(DATA.format("baseline-struct", "test"))[i]
                  for i in range(0, 1540, 97)]

  def test_structure_off_is_unchanged(self):
    plain = make_collator(self.tokenizer, max_length=2048)(self.batch)
    with_struct = make_collator(
        self.tokenizer, max_length=2048,
        structure_fn=structure.GraphStructureFeaturizer(deterministic=True),
    )(self.batch)
    self.assertNotIn("node_ids", plain)
    for key in ("input_ids", "attention_mask", "labels"):
      self.assertTrue(torch.equal(plain[key], with_struct[key]), key)

  def test_token_rows_round_trip(self):
    featurize = structure.GraphStructureFeaturizer(deterministic=True)
    out = make_collator(self.tokenizer, max_length=2048, structure_fn=featurize)(self.batch)
    for i, item in enumerate(self.batch):
      node_ids, labels = out["node_ids"][i], out["labels"][i]
      self.assertTrue((node_ids[labels != -100] == -1).all())
      # Concatenating the digit tokens of each row gives back that node's
      # mentions, e.g. row r -> "13" "13" "13".
      tokens = out["input_ids"][i]
      text = self.tokenizer.decode(tokens[node_ids >= 0])
      expected = "".join(
          item["prompt"][s:e] for s, e, _, _ in featurize(
              item["prompt"][: len(item["prompt"]) - len(item["answer"])]))
      self.assertEqual(text, expected)


if __name__ == "__main__":
  unittest.main()
