"""Index over the MetaQA movie KB (`kb.txt`) and subgraph serialization.

The KB is one graph of ~135k triples, so each question needs a small subgraph carved
out of it. Two strategies are provided. `gold` walks the relation chain named by the
question's qtype, which makes the answer derivable by construction; `retrieved` runs
an undirected k-hop expansion from entities matched in the question text, which adds
realistic retrieval noise and carries no such guarantee.

A pure chain walk emits only on-chain facts, which would make the task trivial -- a
1-hop question would ship a single triple with the answer in it. So the gold subgraph
caps its on-chain content and spends the rest of the triple budget on off-chain
distractors sampled around the chain. Because the chain walk is exhaustive at every
level, those distractors are provably not answers.
"""

PREAMBLE = (
    "In a knowledge graph, (s, p, o) means that entity s is linked to entity o "
    "by relation p.\n"
    "The facts are:\n"
)

# Objects of these relations are literals (years, genres, tags, ...) shared by
# thousands of movies. Expanding through them turns any ball into the whole KB, so
# they are leaves unless they are the seed -- the mirror of KQA Pro's concept fix.
LEAF_RELATIONS = frozenset({
    "release_year", "in_language", "has_genre", "has_tags",
    "has_imdb_rating", "has_imdb_votes",
})

FORWARD = 1

# Cap on nodes kept per chain level; prolific actors would otherwise pull in
# thousands of films whose facts are all discarded by the triple budget anyway.
MAX_FRONTIER = 3000

_STOPWORDS = frozenset(
    "a an the of in on at to for and or is are was were what which who whom whose "
    "when where how many much that this these those with by from as it its does do "
    "did has have had be been being not no yes than then there their between among "
    "movie movies film films person people share same also appear appears star "
    "starred starring acted acting act directed direct director directors write "
    "wrote written writer writers screenwriter screenwriters genre genres type "
    "types language languages year years release released date dates listed "
    "primary main spoken fall under can be described about together co".split()
)

_MAX_NAME_WORDS = 10


def format_edge(edge):
    subject, relation, obj = edge
    return f"({subject}, {relation.replace('_', ' ')}, {obj})"


class KBIndex:
    def __init__(self, kb_path="data/metaqa/kb.txt"):
        self.out, self.inn, self.by_rel = {}, {}, {}
        self.literals = set()
        with open(kb_path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                subject, relation, obj = line.split("|", 2)
                self.out.setdefault(subject, []).append((relation, obj))
                self.inn.setdefault(obj, []).append((relation, subject))
                self.by_rel.setdefault(relation, []).append((subject, relation, obj))
                if relation in LEAF_RELATIONS:
                    self.literals.add(obj)

        self.names = set(self.out) | set(self.inn)
        self.lowered = {}
        for name in self.names:
            self.lowered.setdefault(name.lower(), name)

    # ------------------------------------------------------------------ traversal

    def _step(self, node, relation, direction):
        """Yield (edge, other_node) for one typed hop out of `node`."""
        if direction == FORWARD:
            for rel, obj in self.out.get(node, ()):
                if rel == relation:
                    yield (node, relation, obj), obj
        else:
            for rel, subject in self.inn.get(node, ()):
                if rel == relation:
                    yield (subject, relation, node), subject

    def chain_expand(self, seed, chain, answers):
        """Walk the qtype chain from `seed`, keeping every edge traversed.

        Returns (spine, core, chain_nodes) where `spine` holds one complete
        seed -> answer path per answer reached and `core` is the remaining on-chain
        edges, or None if the chain dies out or reaches none of the answers.

        Reaching only some of the answers is not an error here: the caller decides
        what to do with a partially covered question, and its coverage report would
        be blind to the case if it were dropped at this level.
        """
        if seed not in self.names:
            return None
        levels = [{seed: None}]  # node -> (parent node, edge used to reach it)
        edges = []
        for relation, direction in chain:
            nxt, level_edges = {}, []
            for node in levels[-1]:
                for edge, other in self._step(node, relation, direction):
                    level_edges.append((other, edge))
                    nxt.setdefault(other, (node, edge))
            if not nxt:
                return None
            if len(nxt) > MAX_FRONTIER:
                keep = set(sorted(nxt)[:MAX_FRONTIER])
                keep.update(answer for answer in answers if answer in nxt)
                nxt = {node: nxt[node] for node in keep}
            edges.extend(edge for other, edge in level_edges if other in nxt)
            levels.append(nxt)

        reached = [answer for answer in answers if answer in levels[-1]]
        if not reached:
            return None
        # Answer paths share a prefix near the seed, so they are merged, not concatenated.
        spine = {}
        for answer in reached:
            node = answer
            for level in reversed(levels[1:]):
                parent, edge = level[node]
                spine[edge] = None
                node = parent
        spine = list(spine)

        spine_set = set(spine)
        core = [edge for edge in edges if edge not in spine_set]
        chain_nodes = [node for level in levels for node in level]
        return spine, core, chain_nodes

    def expand(self, seeds, hops, max_nodes):
        """Undirected k-hop ball, with literal nodes treated as leaves."""
        selected, frontier = set(seeds), set(seeds)
        for _ in range(hops):
            if len(selected) >= max_nodes:
                break
            nxt = set()
            for node in frontier:
                if node in self.literals and node not in seeds:
                    continue
                nxt.update(obj for _, obj in self.out.get(node, ()))
                nxt.update(subject for _, subject in self.inn.get(node, ()))
            nxt -= selected
            for node in sorted(nxt):
                if len(selected) >= max_nodes:
                    break
                selected.add(node)
            frontier = nxt & selected
        return selected

    def edges_within(self, nodes):
        return [(subject, relation, obj)
                for subject in nodes
                for relation, obj in self.out.get(subject, ())
                if obj in nodes]

    # ------------------------------------------------------------------- seeding

    def seeds_from_question(self, question, max_seeds=3):
        """Longest matching entity names in the question text, left to right."""
        tokens = question.split()
        seeds, consumed = [], set()
        for size in range(min(_MAX_NAME_WORDS, len(tokens)), 0, -1):
            for start in range(len(tokens) - size + 1):
                span = range(start, start + size)
                if consumed.intersection(span):
                    continue
                gram = " ".join(tokens[start:start + size]).strip("?.,'\"")
                if len(gram) < 3 or gram.lower() in _STOPWORDS:
                    continue
                name = gram if gram in self.names else self.lowered.get(gram.lower())
                if name is None:
                    continue
                seeds.append(name)
                consumed.update(span)
                if len(seeds) >= max_seeds:
                    return seeds
        return seeds

    # --------------------------------------------------------------- distractors

    def _incident(self, node):
        edges = [(node, rel, obj) for rel, obj in self.out.get(node, ())]
        edges += [(subject, rel, node) for rel, subject in self.inn.get(node, ())]
        return edges

    def off_chain_edges(self, chain_nodes, on_chain, answer_relation, budget, rng):
        """Plausible non-answer facts to pad the gold subgraph with.

        Sampled around *every* chain node rather than just the topic, so noise does
        not cluster at the near end of the chain and give the answer's position away,
        and with a quota of edges sharing the answer's relation so the relation name
        alone is not a giveaway.
        """
        picked, seen = [], set(on_chain)

        quota = max(1, budget // 3)
        competitors = self.by_rel.get(answer_relation, ())
        for edge in rng.sample(competitors, min(len(competitors), quota * 4)):
            if edge in seen:
                continue
            seen.add(edge)
            picked.append(edge)
            if len(picked) >= quota:
                break

        # Widen outwards from the chain a ring at a time, shuffling within each ring
        # so nearby facts come first but no single anchor dominates.
        pool, frontier = [], rng.sample(chain_nodes, min(len(chain_nodes), 40))
        visited = set(frontier)
        while frontier and len(pool) < budget:
            ring, nxt = [], []
            for node in frontier:
                for edge in self._incident(node):
                    if edge in seen:
                        continue
                    seen.add(edge)
                    ring.append(edge)
                    other = edge[2] if edge[0] == node else edge[0]
                    if other not in visited and other not in self.literals:
                        visited.add(other)
                        nxt.append(other)
            rng.shuffle(ring)
            pool.extend(ring)
            frontier = nxt[:60]

        picked.extend(pool[:max(0, budget - len(picked))])
        return picked

    # --------------------------------------------------------------- serialization

    def serialize(self, spine, core, filler, max_triples, rng, core_cap=None):
        """Render tiers into prompt text; `spine` is never dropped.

        Returns (text, nnodes, nedges); nnodes counts entities actually mentioned.
        """
        kept = list(spine[:max_triples])
        cap = min(core_cap if core_cap is not None else max_triples, max_triples)
        room = max(0, cap - len(kept))
        kept += rng.sample(core, min(room, len(core)))
        room = max(0, max_triples - len(kept))
        kept += rng.sample(filler, min(room, len(filler)))
        rng.shuffle(kept)

        nodes = {node for subject, _, obj in kept for node in (subject, obj)}
        text = PREAMBLE + "\n".join(format_edge(edge) for edge in kept) + "\n"
        return text, len(nodes), len(kept)

    # ------------------------------------------------------------------- assembly

    def build_subgraph(self, record, strategy, max_triples, max_nodes, rng):
        """Serialize one question's subgraph under the given seeding `strategy`."""
        if strategy == "gold":
            built = self.chain_expand(record["topic"], record["chain"], record["answers"])
            if built is None:
                return None
            spine, core, chain_nodes = built
            # `serialize` would truncate the spine, silently losing an answer's evidence.
            if len(spine) > max_triples:
                return None
            on_chain = set(spine) | set(core)
            # On-chain content is capped so that most of the prompt is distraction,
            # matching the signal-to-noise ratio a k-hop ball produces on KQA Pro.
            core_cap = max(len(spine), max_triples // 3)
            answer_relation = record["chain"][-1][0]
            filler = self.off_chain_edges(chain_nodes, on_chain, answer_relation,
                                          max_triples - len(spine), rng)
            return self.serialize(spine, core, filler, max_triples, rng, core_cap)

        if strategy == "retrieved":
            seeds = self.seeds_from_question(record["question"])
            if not seeds:
                return None
            # Grown until the ball can fill the same triple budget as `gold`, so the
            # two arms differ in *what* they select rather than in how much.
            hops = len(record["chain"])
            selected = self.expand(seeds, hops, max_nodes)
            edges = self.edges_within(selected)
            while len(edges) < max_triples and hops < len(record["chain"]) + 3:
                hops += 1
                selected = self.expand(seeds, hops, max_nodes)
                edges = self.edges_within(selected)
            focus = set(seeds)
            near = [edge for edge in edges if edge[0] in focus or edge[2] in focus]
            far = [edge for edge in edges if edge[0] not in focus and edge[2] not in focus]
            return self.serialize([], near, far, max_triples, rng)

        raise ValueError(f"Unknown seeding strategy: {strategy}")
