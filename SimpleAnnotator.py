import cv2
import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString

drawing = False
start_point = None
end_point = None
current_img = None
img_path = ""
save_dir = ""

original_size = (0, 0)
resized_img = None
scale_x = 1.0
scale_y = 1.0

LABEL = "kettlebell"

def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, end_point, resized_img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp = resized_img.copy()
            cv2.rectangle(temp, start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("Annotator", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        # Scale back to original resolution
        pt1 = (int(start_point[0] * scale_x), int(start_point[1] * scale_y))
        pt2 = (int(end_point[0] * scale_x), int(end_point[1] * scale_y))
        save_annotation(img_path, pt1, pt2)
        print(f"✅ Annotation saved for {os.path.basename(img_path)}")
        cv2.destroyAllWindows()

def save_annotation(img_path, pt1, pt2):
    global save_dir

    filename = os.path.basename(img_path)
    folder = os.path.basename(os.path.dirname(img_path))
    img = cv2.imread(img_path)
    h, w, c = img.shape

    xmin = min(pt1[0], pt2[0])
    xmax = max(pt1[0], pt2[0])
    ymin = min(pt1[1], pt2[1])
    ymax = max(pt1[1], pt2[1])

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = filename
    ET.SubElement(annotation, "path").text = filename

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "roboflow.com"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    ET.SubElement(annotation, "segmented").text = "0"

    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = LABEL
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"
    ET.SubElement(obj, "occluded").text = "0"

    bbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bbox, "xmin").text = str(xmin)
    ET.SubElement(bbox, "xmax").text = str(xmax)
    ET.SubElement(bbox, "ymin").text = str(ymin)
    ET.SubElement(bbox, "ymax").text = str(ymax)

    xml_str = ET.tostring(annotation, encoding='utf-8')
    dom = parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="    ")

    xml_filename = os.path.join(save_dir, filename.replace(".jpg", ".xml"))
    with open(xml_filename, "w") as f:
        f.write(pretty_xml)

def annotate_images(image_folder, output_folder, max_width=800, max_height=720):
    global current_img, img_path, save_dir, resized_img, scale_x, scale_y

    save_dir = output_folder
    os.makedirs(save_dir, exist_ok=True)

    for file in sorted(os.listdir(image_folder)):
        if file.lower().endswith(".jpg"):
            img_path = os.path.join(image_folder, file)
            current_img = cv2.imread(img_path)
            if current_img is None:
                print(f"⚠️ Skipping unreadable image: {file}")
                continue

            h, w = current_img.shape[:2]

            # Calculate scale factors for width and height
            scale_w = max_width / w
            scale_h = max_height / h
            scale = min(scale_w, scale_h, 1.0)  # never upscale

            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_img = cv2.resize(current_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            scale_x = w / new_w
            scale_y = h / new_h

            cv2.namedWindow("Annotator")
            cv2.setMouseCallback("Annotator", draw_rectangle)
            print(f"🔍 Annotating: {file}")
            cv2.imshow("Annotator", resized_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    annotate_images('sliced_kb_videos_into_frames/TGU', 'sliced_kb_videos_into_frames/xml_TGU')
