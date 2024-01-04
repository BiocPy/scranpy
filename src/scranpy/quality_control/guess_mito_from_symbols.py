from typing import Sequence


def guess_mito_from_symbols(symbols: Sequence[str], prefix: str) -> Sequence[int]:
    """Guess mitochondrial genes from their gene symbols.

    Args:
        symbols:
            List of gene symbols.

        prefix:
            Case-insensitive prefix to guess mitochondrial genes.

    Return:
        List of integer indices for the guessed mitochondrial genes.
    """

    prefix = prefix.lower()
    output = []
    for i, symb in enumerate(symbols):
        if symb.lower().startswith(prefix):
            output.append(i)

    return output
