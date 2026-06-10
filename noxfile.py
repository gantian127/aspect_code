import nox


@nox.session
def lint(session):
    """Run linting and formatting via pre-commit."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")
