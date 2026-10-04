import argparse
import collections
import json
import pathlib
import re


LOGICAL = {"and", "or", "not"}


def tokenize(text):
    text = re.sub(r";[^\n]*", "", text)
    return re.findall(r"\(|\)|[^\s()]+", text)


def parse_one(tokens, index=0):
    if tokens[index] != "(":
        return tokens[index], index + 1
    result = []
    index += 1
    while index < len(tokens) and tokens[index] != ")":
        value, index = parse_one(tokens, index)
        result.append(value)
    if index >= len(tokens):
        raise ValueError("unbalanced opening parenthesis")
    return result, index + 1


def section(tree, name):
    for item in tree:
        if isinstance(item, list) and item and str(item[0]).lower() == name.lower():
            return item
    raise ValueError(f"missing section {name}")


def goal_atoms(expression, logical_path=()):
    if not isinstance(expression, list) or not expression:
        raise ValueError(f"invalid goal expression: {expression!r}")
    head = str(expression[0]).lower()
    if head in LOGICAL:
        atoms = []
        for child in expression[1:]:
            atoms.extend(goal_atoms(child, logical_path + (head,)))
        return atoms
    if any(isinstance(item, list) for item in expression[1:]):
        raise ValueError(f"nested non-logical predicate: {expression!r}")
    return [{
        "predicate": str(expression[0]),
        "arguments": [str(item) for item in expression[1:]],
        "logical_path": list(logical_path),
    }]


def parse_bddl(path):
    tokens = tokenize(path.read_text(encoding="utf-8"))
    tree, end = parse_one(tokens)
    if end != len(tokens):
        raise ValueError("trailing tokens after top-level expression")
    language_part = section(tree, ":language")
    goal_part = section(tree, ":goal")
    if len(goal_part) != 2:
        raise ValueError(f"unexpected goal section: {goal_part!r}")
    return {
        "suite": path.parent.name,
        "file": path.name,
        "language": " ".join(str(item) for item in language_part[1:]),
        "goals": goal_atoms(goal_part[1]),
    }


def audit(root):
    records = [parse_bddl(path) for path in sorted(root.glob("*/*.bddl"))]
    predicate_counts = collections.Counter(
        goal["predicate"] for record in records for goal in record["goals"]
    )
    suite_predicates = {}
    suite_task_counts = collections.Counter(record["suite"] for record in records)
    suite_atom_counts = collections.Counter()
    compound_counts = collections.Counter()
    for record in records:
        suite_predicates.setdefault(record["suite"], collections.Counter()).update(
            goal["predicate"] for goal in record["goals"])
        suite_atom_counts[record["suite"]] += len(record["goals"])
        compound_counts[len(record["goals"])] += 1
    return {
        "schema_version": 1,
        "source": str(root),
        "task_count": len(records),
        "suite_task_counts": dict(sorted(suite_task_counts.items())),
        "goal_atom_count": sum(predicate_counts.values()),
        "predicate_counts": dict(sorted(predicate_counts.items())),
        "suite_goal_atom_counts": dict(sorted(suite_atom_counts.items())),
        "suite_predicate_counts": {
            suite: dict(sorted(counts.items()))
            for suite, counts in sorted(suite_predicates.items())
        },
        "goal_atoms_per_task": {
            str(size): count for size, count in sorted(compound_counts.items())
        },
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = audit(args.bddl_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "task_count", "suite_task_counts", "goal_atom_count",
        "predicate_counts", "suite_predicate_counts", "goal_atoms_per_task",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
