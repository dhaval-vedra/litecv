import litecv


def test_import_package():
    assert hasattr(litecv, '__version__')
    assert litecv.__version__ == '0.1.0'


def test_new_image():
    img = litecv.new_image(100, 100, color='white')
    assert img.width == 100
    assert img.height == 100
    assert img.save is not None
