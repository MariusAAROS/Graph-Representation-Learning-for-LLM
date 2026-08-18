"""Index over the KQA Pro `kb.json` and subgraph serialization.

The KQA Pro KB is a single graph of ~17k entities, so — unlike GraphQA, where the
whole graph fits in the prompt — each question needs a small subgraph carved out
of it. Two seeding strategies are provided: `gold` (entities named by the KoPL
program, so the answer is guaranteed derivable) and `retrieved` (entities matched
against the question text, which adds realistic retrieval noise).
"""

import json

PREAMBLE = (
    "In a knowledge graph, (s, p, o) means that entity s is linked to entity o "
    "by relation p, and (s, k, v) means that entity s has attribute k with value "
    "v. Facts may carry qualifiers, given in brackets.\n"
    "The facts are:\n"
)

VALUE_FILTERS = frozenset({"FilterStr", "FilterNum", "FilterYear", "FilterDate"})

_STOPWORDS = frozenset(
    "a an the of in on at to for and or is are was were what which who whom whose "
    "when where how many much that this these those with by from as it its does do "
    "did has have had be been being not no yes than then there their between among "
    "whether same both all any more most less least number one".split()
)


def format_value(value):
    """Render a KQA Pro typed value ({type, value, unit}) as plain text."""
    kind, raw = value.get("type"), value.get("value")
    if kind == "quantity":
        unit = value.get("unit") or "1"
        text = f"{raw:g}" if isinstance(raw, float) else str(raw)
        return text if unit == "1" else f"{text} {unit}"
    if kind == "year":
        return str(int(raw))
    return str(raw)


def format_qualifiers(qualifiers):
    if not qualifiers:
        return ""
    parts = [f"{key}: {format_value(value)}"
             for key, values in sorted(qualifiers.items()) for value in values]
    return f" [{'; '.join(parts)}]" if parts else ""


def load_kopl_engine(kb_path="data/kqapro/kb.json"):
    """Build a KoPLEngine over a fresh copy of the KB.

    The `kopl` package expects the KoPL-toolkit schema, while the KQA Pro release
    uses `instanceOf` on concepts and `predicate` on relations; rename both. The KB
    is re-read from disk because `kopl.data.KB` pops keys off the dict it is given.
    """
    from kopl.kopl import KoPLEngine

    with open(kb_path) as f:
        kb = json.load(f)
    for concept in kb["concepts"].values():
        concept["subclassOf"] = concept.pop("instanceOf")
    for entity in kb["entities"].values():
        for rel in entity["relations"]:
            rel["relation"] = rel.pop("predicate")
    return KoPLEngine(kb)


def program_terms(program):
    """Every string the KoPL program mentions (entity/concept names, keys, values)."""
    return {term for inputs in program["inputs"] for term in inputs}


class KBIndex:
    def __init__(self, kb_path="data/kqapro/kb.json"):
        with open(kb_path) as f:
            kb = json.load(f)
        self.kb = kb
        self.entities = kb["entities"]
        self.concepts = kb["concepts"]

        # Relations may point at concepts as well as entities, so concepts are
        # nodes of the graph too -- leaf ones, since they carry no relations.
        self.node_names = {nid: node["name"]
                           for nid, node in list(self.concepts.items()) + list(self.entities.items())}
        self.name_to_ids = {}
        for nid, name in self.node_names.items():
            self.name_to_ids.setdefault(name, []).append(nid)

        self._neighbors = {
            eid: {rel["object"] for rel in entity["relations"]
                  if rel["object"] in self.node_names}
            for eid, entity in self.entities.items()
        }
        # Programs built on FindAll identify their target by attribute value rather
        # than by name, so values need to be resolvable back to entities too.
        self.value_to_ids = {}
        for eid, entity in self.entities.items():
            for attr in entity["attributes"]:
                self.value_to_ids.setdefault(format_value(attr["value"]), []).append(eid)
        # Lowercased surface forms, longest first, so retrieval prefers specific names.
        self._surface_forms = sorted(
            ((name.lower(), name) for name in self.name_to_ids),
            key=lambda pair: -len(pair[0]),
        )

    def concept_name(self, cid):
        concept = self.concepts.get(cid)
        return concept["name"] if concept else cid

    def ids_for_name(self, name):
        return self.name_to_ids.get(name, [])

    def expand(self, seed_ids, hops, max_nodes):
        """Breadth-first k-hop neighbourhood, truncated at `max_nodes` entities."""
        selected = set(seed_ids)
        frontier = set(seed_ids)
        for _ in range(hops):
            if len(selected) >= max_nodes:
                break
            nxt = set()
            for eid in frontier:
                nxt |= self._neighbors.get(eid, set())
            nxt -= selected
            for eid in sorted(nxt):
                if len(selected) >= max_nodes:
                    break
                selected.add(eid)
            frontier = nxt & selected
        return selected

    def seeds_from_program(self, program, answer=None):
        """Gold seeding: the entities the program anchors on, plus the answer entity.

        Anchors are `Find` arguments (entity names) and the values of attribute
        filters, which is how `FindAll`-rooted programs pin down their target.
        """
        seeds = []
        for fn, inputs in zip(program["function"], program["inputs"]):
            if fn == "Find" and inputs:
                seeds.extend(self.ids_for_name(inputs[0]))
            elif fn in VALUE_FILTERS and len(inputs) >= 2:
                seeds.extend(self.value_to_ids.get(inputs[1], []))
        if answer:
            seeds.extend(self.ids_for_name(answer))
            seeds.extend(self.value_to_ids.get(answer, []))
        return seeds

    def seeds_from_question(self, question, max_seeds=4):
        """Retrieval seeding: longest entity names occurring in the question text."""
        haystack = question.lower()
        seeds, consumed = [], []
        for lowered, name in self._surface_forms:
            if len(lowered) < 3 or lowered in _STOPWORDS:
                continue
            start = haystack.find(lowered)
            if start < 0:
                continue
            end = start + len(lowered)
            if any(start < c_end and c_start < end for c_start, c_end in consumed):
                continue
            consumed.append((start, end))
            seeds.extend(self.ids_for_name(name))
            if len(consumed) >= max_seeds:
                break
        return seeds

    def _triples_for(self, eid, selected, terms, focus):
        """Yield (is_core, text) facts for one entity, restricted to `selected`.

        A fact is core when it plausibly carries the answer: it touches one of the
        question's focus entities and mentions something the program names, or it
        links two focus entities. Everything else is a distractor.
        """
        entity = self.entities[eid]
        name = entity["name"]
        in_focus = eid in focus
        for cid in entity["instanceOf"]:
            concept = self.concept_name(cid)
            yield in_focus and concept in terms, f"({name}, instance of, {concept})"
        for attr in entity["attributes"]:
            value = format_value(attr["value"])
            named_key = attr["key"] in terms
            relevant = (named_key or value in terms
                        or any(qk in terms for qk in attr["qualifiers"]))
            # A program-named key stays core even off-focus: multi-hop questions ask
            # for an attribute of an entity reached from the focus, not of it.
            yield relevant and (in_focus or named_key), (
                f"({name}, {attr['key']}, {value})"
                f"{format_qualifiers(attr['qualifiers'])}")
        for rel in entity["relations"]:
            if rel["direction"] != "forward" or rel["object"] not in selected:
                continue
            obj = self.node_names[rel["object"]]
            touches_focus = in_focus or rel["object"] in focus
            core = touches_focus and (rel["predicate"] in terms
                                      or obj in terms or name in terms)
            yield core, (f"({name}, {rel['predicate']}, {obj})"
                         f"{format_qualifiers(rel['qualifiers'])}")

    def serialize(self, selected, terms, focus, answer, max_triples, rng):
        """Render `selected` entities as text, keeping program-relevant facts first.

        Facts stating the answer are always kept, so a `gold` subgraph is guaranteed
        to support its question no matter how tight the triple budget is.
        Returns (text, nnodes, nedges); nnodes counts entities actually mentioned.
        """
        must, core, distractors = [], [], []
        for eid in sorted(selected):
            if eid not in self.entities:  # concepts are leaves; they emit no facts
                continue
            for is_core, text in self._triples_for(eid, selected, terms, focus):
                if answer and answer in text:
                    # Common answers (years, short names) occur in hundreds of facts;
                    # prefer the ones actually anchored on the question's entities.
                    must.append((0 if eid in focus else 1 if is_core else 2, eid, text))
                elif is_core:
                    core.append((eid, text))
                else:
                    distractors.append((eid, text))

        must.sort(key=lambda item: item[0])
        kept = [(eid, text) for _, eid, text in must[:max(1, max_triples // 3)]]
        triples = kept + core[:max(0, max_triples - len(kept))]
        budget = max_triples - len(triples)
        if budget > 0:
            triples += rng.sample(distractors, min(budget, len(distractors)))
        rng.shuffle(triples)

        nnodes = len({eid for eid, _ in triples})
        text = PREAMBLE + "\n".join(t for _, t in triples) + "\n"
        return text, nnodes, len(triples)

    def build_subgraph(self, record, strategy, hops, max_triples, max_nodes, rng):
        """Serialize one question's subgraph under the given seeding `strategy`."""
        program, terms = record["program"], program_terms(record["program"])
        answer = record.get("answer")
        if strategy == "gold":
            seeds = self.seeds_from_program(program, answer)
        elif strategy == "retrieved":
            # Retrieval must not peek at the answer, so its facts are not forced in.
            seeds, answer = self.seeds_from_question(record["question"]), None
        else:
            raise ValueError(f"Unknown seeding strategy: {strategy}")
        if not seeds:
            return None
        focus = set(seeds)
        selected = self.expand(seeds, hops=hops, max_nodes=max_nodes)
        return self.serialize(selected, terms, focus, answer, max_triples, rng)
