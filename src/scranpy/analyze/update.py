import copy

__author__ = "ltla, jkanche"
__copyright__ = "ltla"
__license__ = "MIT"


def update(options, **kwargs):
    """Convenience function to update the settings on an ``*Options`` object.

    Args:
        options:
            Any of the ``*Options`` object.

        kwargs:
            Key-value pairs of settings to replace.

    Results:
        A copy of ``options`` with replaced settings.
        Note that the input ``options`` itself is unchanged.
    """
    output = copy.copy(options)
    for k, v in kwargs.items():
        setattr(output, k, v)
    return output
