from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from project_paths import project_file,source_files


class ProjectPathTests(unittest.TestCase):
    def test_moved_source_and_tests_resolve(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'research').mkdir();(root/'tests').mkdir()
            for name in ('research/controller.py','tests/test_controller.py'):
                (root/name).write_text('# fixture\n')
            self.assertEqual(project_file(root,'controller.py'),root/'research/controller.py')
            self.assertEqual(project_file(root,'test_controller.py'),root/'tests/test_controller.py')
            self.assertEqual(source_files(root),sorted([root/'research/controller.py',root/'tests/test_controller.py']))

    def test_existing_and_missing_paths_keep_original_behavior(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'controller.py').write_text('# original\n')
            self.assertEqual(project_file(root,'controller.py'),root/'controller.py')
            self.assertEqual(project_file(root,'results/data.json'),root/'results/data.json')
            missing=project_file(root,'missing.py')
            self.assertEqual(missing,root/'missing.py')
            with self.assertRaises(FileNotFoundError):missing.read_bytes()
