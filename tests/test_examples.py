import pathlib
import py_compile


def test_example_scripts_compile():
    root = pathlib.Path(__file__).resolve().parent.parent / 'examples'
    for script in root.glob('*.py'):
        py_compile.compile(str(script), doraise=True)


def test_example_scripts_exist():
    root = pathlib.Path(__file__).resolve().parent.parent / 'examples'
    expected = {
        'demo.py',
        'basic_image.py',
        'filters_demo.py',
        'camera_app.py',
        'video_demo.py',
        'object_detection.py',
        'utilities.py',
    }
    found = {p.name for p in root.glob('*.py')}
    assert expected.issubset(found)
