import textwrap


def create_text_wrapper(width=80):

    def f(text, indent_level=0):
        return print_wrap(text, indent_level, width)

    return f


def print_wrap(text, indent_level=0, width=80):
    """
    Wrap texts
    """
    space = '   '
    return textwrap.fill(text, width,
                         initial_indent=indent_level*space,
                         subsequent_indent=(indent_level+1)*space)


def remove_duplicates(seq):
    """
    Remove duplicates in a list and keep the order
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def build_embedding_dictionary(lookup_matrix, inv_vocab):
    """
    lookup_matrix is a 2D numpy array of shape (vocab_size, size_embedding)
    inv_vocab is a dictionary (id_word, "str_word")

    Returns a dictionary ("str_word", embedding_word)
    """
    embedding_dictionary = {}
    for word_id in range(lookup_matrix.shape[0]):
        embedding_dictionary[inv_vocab[word_id]] = lookup_matrix[word_id]

    return embedding_dictionary
