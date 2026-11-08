import ast

import pytest

from payrules.dslmine import apply_templates, exact_duplicates, load_corpus, mine, round_trips

FOO = """
def make_{a}(value):
    total = value * {n}
    if total > 10:
        print("large {a}", total)
    else:
        print("small {a}", total)
    result = [total, value, total + value]
    return result
"""


def corpus(*texts, min_size=20):
    return load_corpus([(f"f{i}.py", text) for i, text in enumerate(texts)], min_size=min_size)


def test_family_becomes_a_template_with_typed_parameters():
    templates, coverage = mine(corpus(FOO.format(a="a", n=2), FOO.format(a="b", n=3)), 10)
    assert len(templates) == 1
    assert coverage.rewrites == 2
    assert "int" in templates[0].parameter_types and "str" in templates[0].parameter_types


def test_every_mined_rewrite_rebuilds_the_exact_tree():
    data = corpus(FOO.format(a="a", n=2), FOO.format(a="b", n=3), FOO.format(a="c", n=4))
    templates, _ = mine(data, 10)
    instances = data.families[templates[0].key]
    assert all(round_trips(templates[0], i) for i in instances)


def test_leaves_that_always_match_share_one_parameter():
    code = "def f_{a}(x):\n    y = x + {n}\n    z = y * {n}\n    w = [x, y, z, {n}]\n    return w\n"
    templates, _ = mine(corpus(code.format(a="p", n=1), code.format(a="q", n=2)), 10)
    assert templates[0].parameter_types.count("int") == 1


def test_arguments_are_type_checked():
    templates, _ = mine(corpus(FOO.format(a="a", n=2), FOO.format(a="b", n=3)), 10)
    template = templates[0]
    arguments = template.bind(next(iter(corpus(FOO.format(a="z", n=9)).families.values()))[0])
    template.expand(arguments)
    wrong = tuple(object() for _ in arguments)
    with pytest.raises(TypeError):
        template.expand(wrong)
    with pytest.raises(TypeError):
        template.expand(arguments[:-1])


def test_different_shapes_do_not_unify():
    other = FOO.format(a="b", n=3).replace("    return result", "    print(result)\n    return result")
    templates, _ = mine(corpus(FOO.format(a="a", n=2), other, min_size=40), 10)
    assert templates == []


def test_nested_statements_are_not_counted_twice():
    data = corpus(FOO.format(a="a", n=2), FOO.format(a="b", n=3), min_size=5)
    _, coverage = mine(data, 100)
    for lines in coverage.claimed.values():
        assert len(lines) <= len(FOO.strip().splitlines())
    total = sum(1 for _ in ast.walk(ast.parse(FOO.format(a="a", n=2)))) * 2
    assert coverage.nodes <= total


def test_templates_apply_to_unseen_code_but_copy_paste_detection_does_not():
    train = corpus(FOO.format(a="a", n=2), FOO.format(a="b", n=3), FOO.format(a="c", n=2))
    templates, _ = mine(train, 10)
    # A fixed leaf must match, so unseen code differs only where the family differed.
    test = corpus(FOO.format(a="z", n=9))
    assert apply_templates(templates, test).rewrites == 1
    assert exact_duplicates(train, test)[0].rewrites == 0


def test_copy_paste_baseline_respects_its_size_limit():
    train = corpus(FOO.format(a="a", n=2), FOO.format(a="a", n=2), FOO.format(a="b", n=3))
    limited, entries = exact_duplicates(train, corpus(FOO.format(a="b", n=3)), max_entries=1)
    assert entries == 1 and limited.rewrites == 0
    assert exact_duplicates(train, corpus(FOO.format(a="a", n=2)), max_entries=1)[0].rewrites == 1


def test_a_fixed_leaf_that_differs_blocks_the_rewrite():
    train = corpus(FOO.format(a="a", n=2), FOO.format(a="b", n=3))
    templates, _ = mine(train, 10)
    changed = FOO.format(a="z", n=9).replace("total > 10", "total > 99")
    assert apply_templates(templates, corpus(changed)).rewrites == 0


def test_bad_files_are_skipped_and_counted():
    data = load_corpus([("bad.py", "def ("), ("ok.py", "x = 1\n")])
    assert data.skipped_files == 1 and data.files == 1
