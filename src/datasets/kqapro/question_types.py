"""Derive a single question-type label from a KQA Pro KoPL program.

KQA Pro ships no question-type field, and the reasoning skills it exercises are not
mutually exclusive (one question is routinely Count *and* Multihop *and* Qualifier).
These labels are therefore a reporting bucket that feeds the per-task
`val/exact_match_{task}` breakdown -- not a train/test partition. Precedence runs
from the most distinctive skill to the least.
"""

VERIFY = frozenset({"VerifyStr", "VerifyNum", "VerifyYear", "VerifyDate"})
COMPARISON = frozenset({"SelectBetween", "SelectAmong"})
LOGICAL = frozenset({"And", "Or"})
QUALIFIER = frozenset({
    "QFilterStr", "QFilterNum", "QFilterYear", "QFilterDate",
    "QueryAttrQualifier", "QueryRelationQualifier", "QueryAttrUnderCondition",
})

TASKS = ["Count", "Verify", "Comparison", "Qualifier", "Multihop", "Logical", "Basic"]


def derive_task(program):
    functions = program["function"]
    present = set(functions)
    if "Count" in present:
        return "Count"
    if present & VERIFY:
        return "Verify"
    if present & COMPARISON:
        return "Comparison"
    if present & QUALIFIER:
        return "Qualifier"
    if functions.count("Relate") >= 2:
        return "Multihop"
    if present & LOGICAL:
        return "Logical"
    return "Basic"


def program_length(program):
    """Number of KoPL steps; an a-priori difficulty signal for the curriculum."""
    return len(program["function"])
