"""
db subpackage

Houses rysy_dl's packaged default circuit database. Living inside the package
(rather than as a sibling of src/) means it's covered by the existing PrairieLearn
bind-mount of src/rysy_dl and by any future pip/PyPI package-data declaration,
with no extra wiring needed in either case.
"""

from pathlib import Path

DB_DIR = Path(__file__).parent
DEFAULT_DB_PATH = DB_DIR / 'rysy_dag_only_4-7.sqlite'


def resolve_db_path(db_path):
    """
    Resolve a db_path argument, honoring the '@' prefix convention: a string of the form
    '@name.sqlite' refers to 'name.sqlite' packaged inside this db subpackage, rather than a
    path relative to the caller's current working directory. Anything else -- a plain string,
    a Path, or None -- is returned unchanged.

    :param db_path: The db_path value passed to `CircuitDatabase` or `create_database`.
    :returns: A resolved path (str or Path) with the '@' prefix, if any, replaced by `DB_DIR`.
    """
    if isinstance(db_path, str) and db_path.startswith('@'):
        return DB_DIR / db_path[1:]
    return db_path
