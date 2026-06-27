import ragnarok

def test_package_has_version():
    assert isinstance(ragnarok.__version__, str)
    assert ragnarok.__version__
