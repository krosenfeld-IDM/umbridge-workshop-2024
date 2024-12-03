"""
https://panel.holoviz.org/how_to/test/pytest.html
"""
import pytest

from app import App

@pytest.fixture
def app():
    return App(run_delay=0.001, load_delay=0.001)

def test_constructor(app):
    """Tests default values of App"""
    # Then
    assert app.run == False # noqa: E712
    assert app.status == "No runs yet"
    assert app.runs == 0

def test_run(app):
    """Tests behaviour when Run button is clicked once"""
    # When
    app.param.trigger('run')
    # Then
    assert app.runs == 1
    assert app.status.startswith("Finished run 1 in")

def test_run_twice(app):
    """Tests behaviour when Run button is clicked twice"""
    # When
    app.param.trigger('run')
    app.param.trigger('run')
    # Then
    assert app.runs == 2
    assert app.status.startswith("Finished run 2 in")