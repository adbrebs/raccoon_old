import textwrap


def print_wrap(str, indent_level, width=80):
    """
    Wrap texts
    """
    space = '   '
    return textwrap.fill(str, width,
                         initial_indent=indent_level*space,
                         subsequent_indent=(indent_level+1)*space)


def remove_duplicates(seq):
    """
    Remove duplicates in a list and keep the order
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
