import os

from git import Repo


repo = Repo(os.getcwd(), search_parent_directories=True)
git_root = repo.git.rev_parse("--show-toplevel")

ASSETS_ROOT = os.path.join(git_root, 'assets')