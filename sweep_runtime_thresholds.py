import os, glob, argparse, math, xml.etree.ElementTree as ET
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow import convert_to_tensor
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)


model = tf.saved_model.load("barbell_models/efficientdet_lite0_2025-10-22_15-17-32/saved_model")
detect_fn = model.signatures['serving_default']
# -----------------------------
# Pascal VOC parsing
# -----------------------------
def parse_voc(xml_path, label_filter=None):
    root = ET.parse(xml_path).getroot()
    W = int(float(root.findtext("size/width")))
    H = int(float(root.findtext("size/height")))
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if label_filter and name not in label_filter:
            continue
        bb = obj.find("bndbox")
        xmin = int(float(bb.findtext("xmin")));
        ymin = int(float(bb.findtext("ymin")))
        xmax = int(float(bb.findtext("xmax")));
        ymax = int(float(bb.findtext("ymax")))
        xmin = max(0, min(xmin, W - 1));
        xmax = max(0, min(xmax, W - 1))
        ymin = max(0, min(ymin, H - 1));
        ymax = max(0, min(ymax, H - 1))
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
    return W, H, np.array(boxes, dtype=np.float32)


def load_dataset(folder, img_exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp"), label_filter={"barbell"}):
    xmls = sorted(glob.glob(os.path.join(folder, "*.xml")))
    samples = []
    for xp in xmls:
        stem = os.path.splitext(os.path.basename(xp))[0]
        img_path = None
        for ext in img_exts:
            p = os.path.join(folder, stem + ext)
            if os.path.exists(p):
                img_path = p;
                break
        if img_path is None:
            continue
        W, H, gts = parse_voc(xp, label_filter)
        samples.append((img_path, gts, (W, H)))
    return samples


# -----------------------------
# IoU + NMS
# -----------------------------
def iou_xyxy(a, b):
    # a: [N,4], b: [M,4]
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(boxes, scores, iou_thresh):
    idxs = np.argsort(-scores)
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1: break
        rest = idxs[1:]
        ious = iou_xyxy(boxes[i:i + 1], boxes[rest]).ravel()
        idxs = rest[ious < iou_thresh]
    return np.array(keep, dtype=np.int32)


# -----------------------------
# TFLite runner
# -----------------------------
def infer_gpu(image):
    # Convert to RGB float tensor
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = convert_to_tensor(img_rgb[None, ...], dtype=tf.uint8)

    # Run the model on GPU (uses CUDA automatically)
    outputs = detect_fn(tensor)

    # Your specific output mapping
    boxes = outputs["output_0"][0].numpy()  # [N,4] ymin,xmin,ymax,xmax normalized
    scores = outputs["output_2"][0].numpy()  # [N]
    # classes = outputs["output_1"][0].numpy()  # unused here
    # num_det = outputs["output_3"][0].numpy()  # optional

    # Convert to absolute XYXY
    H0, W0 = image.shape[:2]
    ymin, xmin, ymax, xmax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy = np.stack([xmin * W0, ymin * H0, xmax * W0, ymax * H0], axis=1).astype(np.float32)

    return xyxy, scores


class TFLiteDetector:
    def __init__(self, model_path):
        self.interp = tf.lite.Interpreter(model_path=model_path, num_threads=os.cpu_count() or 4)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outs = self.interp.get_output_details()
        _, self.H, self.W, _ = self.inp["shape"]
        self.input_dtype = self.inp["dtype"]

        # Manual mapping for Model Maker EfficientDet-Lite exports
        # Order: [count, classes, scores, boxes] or slight variant — check names
        name_to_idx = {od["name"]: i for i, od in enumerate(self.outs)}

        # Your model layout:
        # [0] = count (StatefulPartitionedCall:0)
        # [1] = classes (StatefulPartitionedCall:1)
        # [2] = scores (StatefulPartitionedCall:2)
        # [3] = boxes (StatefulPartitionedCall:3)
        self.idx_count = name_to_idx.get("StatefulPartitionedCall:0")
        self.idx_classes = name_to_idx.get("StatefulPartitionedCall:1")
        self.idx_scores = name_to_idx.get("StatefulPartitionedCall:2")
        self.idx_boxes = name_to_idx.get("StatefulPartitionedCall:3")

        assert all(x is not None for x in [
            self.idx_boxes, self.idx_scores, self.idx_classes, self.idx_count
        ]), f"Unexpected TFLite output layout: {name_to_idx}"

    def preprocess(self, img_bgr):
        img = cv2.resize(img_bgr, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if self.input_dtype == np.uint8:
            return img[np.newaxis, ...]
        # float32 pathway
        img = img.astype(np.float32) / 255.0
        return img[np.newaxis, ...]


# -----------------------------
# Matching & metrics (single-class)
# -----------------------------
def match_image(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    # Greedy match by score
    order = np.argsort(-pred_scores)
    used_gt = np.zeros((gt_boxes.shape[0],), dtype=bool)
    tp = 0;
    fp = 0
    for i in order:
        if pred_scores[i] <= 0:
            continue
        if gt_boxes.size == 0:
            fp += 1;
            continue
        ious = iou_xyxy(pred_boxes[i:i + 1], gt_boxes).ravel()
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not used_gt[j]:
            tp += 1
            used_gt[j] = True
        else:
            fp += 1
    fn = int(np.sum(~used_gt))
    return tp, fp, fn


def precision_recall_f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# -----------------------------
# Sweep
# -----------------------------
def sweep(model_path, data_dir, sample_frac=1.0, iou_match=0.5,
          score_list=(0.55, 0.60, 0.65, 0.70, 0.75),
          nms_iou_list=(0.40, 0.45, 0.50, 0.55, 0.60),
          label_filter={"barbell"}):
    data = load_dataset(data_dir, label_filter=label_filter)
    if sample_frac < 1.0:
        k = max(1, int(len(data) * sample_frac))
        data = data[:k]

    det = TFLiteDetector(model_path)
    results = []

    for thr in score_list:
        for nms_iou in nms_iou_list:
            TP = FP = FN = 0
            for img_path, gts, (W, H) in tqdm(data, desc=f"thr={thr:.2f}, nms={nms_iou:.2f}", leave=False):
                img = cv2.imread(img_path)
                if img is None:
                    continue
                boxes, scores = det.infer(img, (W, H))

                # 1) filter by score (only upwards; underlying model already applied a base threshold)
                m = scores >= thr
                boxes = boxes[m];
                scores_sel = scores[m]
                if boxes.size == 0:
                    FN += len(gts);
                    continue

                # 2) extra NMS
                keep = nms(boxes, scores_sel, nms_iou)
                boxes = boxes[keep];
                scores_sel = scores_sel[keep]

                tp, fp, fn = match_image(boxes, scores_sel, gts, iou_thr=iou_match)
                TP += tp;
                FP += fp;
                FN += fn

            prec, rec, f1 = precision_recall_f1(TP, FP, FN)
            results.append((thr, nms_iou, prec, rec, f1, TP, FP, FN))

    # sort by F1 then precision
    results.sort(key=lambda x: (x[4], x[2]), reverse=True)
    return results


# -----------------------------
# CLI
# -----------------------------
def main():
    # model_path = r"barbell_models/efficientdet_lite0_2025-10-22_15-17-32/__BARBELL__efficientdet_lite0_2025-10-22_15-17-32.tflite"
    # interpreter = tf.lite.Interpreter(model_path=model_path)
    # interpreter.allocate_tensors()
    # outputs = interpreter.get_output_details()
    #
    # print("\nOutput tensor details:")
    # for i, od in enumerate(outputs):
    #     print(f"[{i}] name={od['name']}, shape={od['shape']}, dtype={od['dtype']}")

    ap = argparse.ArgumentParser(
        description="Sweep score threshold & NMS IoU on a TFLite detector (single-class, VOC).")
    ap.add_argument("--model", required=True, help="Path to .tflite")
    ap.add_argument("--data", required=True, help="Folder with images + VOC XML")
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--iou-match", type=float, default=0.5, help="IoU for GT matching (0.5 by default)")
    ap.add_argument("--scores", type=str, default="0.55,0.6,0.65,0.7,0.75")
    ap.add_argument("--nms-ious", type=str, default="0.40,0.45,0.50,0.55,0.60")
    args = ap.parse_args()

    score_list = [float(x) for x in args.scores.split(",")]
    nms_iou_list = [float(x) for x in args.nms_ious.split(",")]

    results = sweep(args.model, args.data, args.sample_frac, args.iou_match, score_list, nms_iou_list)

    print("\n=== Top results (F1 @ IoU={:.2f}) ===".format(args.iou_match))
    print(" score_thr | nms_iou | precision | recall |   F1   | TP  FP  FN")
    for i, (thr, nms_iou, p, r, f1, TP, FP, FN) in enumerate(results[:10]):
        print(f"   {thr:6.2f} |  {nms_iou:6.2f} |   {p:7.3f} | {r:6.3f} | {f1:6.3f} | {TP:4d} {FP:4d} {FN:4d}")

    best = results[0]
    print("\nBest pair -> score_thresh: {:.2f}, extra_nms_iou: {:.2f} (F1={:.3f}, P={:.3f}, R={:.3f})"
          .format(best[0], best[1], best[4], best[2], best[3]))

    # model = tf.saved_model.load("barbell_models/efficientdet_lite0_2025-10-22_15-17-32/saved_model")
    # detect_fn = model.signatures['serving_default']
    #
    # print("=== Output keys of serving_default ===")
    # for key, val in detect_fn.structured_outputs.items():
    #     print(f"{key}: shape={val.shape}, dtype={val.dtype}")


if __name__ == "__main__":
    main()
