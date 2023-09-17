from scranpy import guess_mito_from_symbols


def test_guess_mito_from_symbols():
    out = guess_mito_from_symbols(["asdasd", "mt-asdas", "sadasd", "MT-asdasd"], "mt-")
    assert out == [1, 3]
