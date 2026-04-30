import sys
import types
import unittest


def _install_import_stubs():
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    pil_module = sys.modules.setdefault("PIL", types.ModuleType("PIL"))
    pil_module.__path__ = []
    sys.modules.setdefault("PIL.Image", types.ModuleType("PIL.Image"))
    sys.modules.setdefault("PIL.ImageTk", types.ModuleType("PIL.ImageTk"))


_install_import_stubs()

from PreviewValidateTab import PreviewValidateTab


class _ImageStub:
    shape = (2160, 3840, 3)


class PreviewValidateTabXmlTests(unittest.TestCase):
    def test_build_reannotation_xml_uses_display_dimensions_for_box_scaling(self):
        tab = PreviewValidateTab.__new__(PreviewValidateTab)
        tab.reannotation_image_path = r"C:\dataset\sample.jpg"
        tab.reannotation_image = _ImageStub()
        tab.reannotation_scale = 0.2
        tab.reannotation_display_size = (799, 450)
        tab.reannotation_annotations = [
            {
                "x1": 250,
                "y1": 100,
                "x2": 400,
                "y2": 250,
                "label": "dumbbell",
            }
        ]

        root, valid_annotations, filename = tab._build_reannotation_xml_root()

        self.assertEqual(valid_annotations, 1)
        self.assertEqual(filename, "sample.jpg")

        size = root.find("size")
        self.assertEqual(size.findtext("width"), "3840")
        self.assertEqual(size.findtext("height"), "2160")

        bbox = root.find("object").find("bndbox")
        self.assertEqual(bbox.findtext("xmin"), str(int(round(250 * 3840 / 799))))
        self.assertEqual(bbox.findtext("xmax"), str(int(round(400 * 3840 / 799))))
        self.assertEqual(bbox.findtext("ymin"), str(int(round(100 * 2160 / 450))))
        self.assertEqual(bbox.findtext("ymax"), str(int(round(250 * 2160 / 450))))


if __name__ == "__main__":
    unittest.main()
