"""Turn a MetaQA qtype string into the KB relation chain it traverses.

Every MetaQA question is a chain walk over the movie KB, and `qa_*_qtype.txt` names
that chain segment by segment (`movie_to_actor_to_movie_to_year`). Mapping each
consecutive pair of segments to a KB edge gives the exact reasoning path, which is
what lets the gold subgraph be carved by following it rather than by guessing.

Unlike KQA Pro, the task label is not derived from the reasoning skill: MetaQA's ~49
qtypes are all the same skill at different depths, so `task` is the hop count and the
qtype only supplies the chain.
"""

FORWARD, BACKWARD = 1, -1

# (from segment, to segment) -> (KB relation, direction). Backward steps traverse the
# triple against its stored direction; the KB only stores movie-as-subject facts.
PAIR_TO_EDGE = {
    ("movie", "actor"): ("starred_actors", FORWARD),
    ("actor", "movie"): ("starred_actors", BACKWARD),
    ("movie", "director"): ("directed_by", FORWARD),
    ("director", "movie"): ("directed_by", BACKWARD),
    ("movie", "writer"): ("written_by", FORWARD),
    ("writer", "movie"): ("written_by", BACKWARD),
    ("movie", "genre"): ("has_genre", FORWARD),
    ("movie", "language"): ("in_language", FORWARD),
    ("movie", "year"): ("release_year", FORWARD),
    ("movie", "tags"): ("has_tags", FORWARD),
    ("tag", "movie"): ("has_tags", BACKWARD),
    ("movie", "imdbrating"): ("has_imdb_rating", FORWARD),
    ("movie", "imdbvotes"): ("has_imdb_votes", FORWARD),
}


def qtype_chain(qtype):
    """['movie', 'actor', 'movie'] -> [(starred_actors, +1), (starred_actors, -1)]."""
    segments = qtype.strip().split("_to_")
    chain = []
    for pair in zip(segments, segments[1:]):
        if pair not in PAIR_TO_EDGE:
            raise ValueError(f"No KB edge for segment pair {pair} in qtype {qtype!r}")
        chain.append(PAIR_TO_EDGE[pair])
    return chain


def derive_task(nsteps):
    return f"{nsteps}hop"
