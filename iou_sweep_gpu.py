# iou_sweep_gpu.py
# Sweep score threshold & extra NMS IoU using a TensorFlow SavedModel (GPU/CUDA).
# Robust to SavedModel output naming + normalized/pixel coords. Single-class VOC.

import os, glob, argparse, xml.etree.ElementTree as ET
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow import convert_to_tensor

# -----------------------------
# GPU setup
# -----------------------------
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"✅ Using {len(gpus)} GPU(s) with memory growth.")
        except Exception as e:
            print("⚠️ Could not set memory growth:", e)
    else:
        print("⚠️ No GPU detected. Running on CPU.")

# -----------------------------
# Pascal VOC parsing
# -----------------------------
def parse_voc(xml_path, label_filter=None):
    root = ET.parse(xml_path).getroot()
    W = int(float(root.findtext("size/width")));  H = int(float(root.findtext("size/height")))
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if label_filter and name not in label_filter:
            continue
        bb = obj.find("bndbox")
        xmin = int(float(bb.findtext("xmin"))); ymin = int(float(bb.findtext("ymin")))
        xmax = int(float(bb.findtext("xmax"))); ymax = int(float(bb.findtext("ymax")))
        xmin = max(0, min(xmin, W-1)); xmax = max(0, min(xmax, W-1))
        ymin = max(0, min(ymin, H-1)); ymax = max(0, min(ymax, H-1))
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
    return W, H, np.array(boxes, dtype=np.float32)

def load_dataset(folder, img_exts=(".jpg",".jpeg",".png",".bmp",".webp"), label_filter={"barbell"}):
    xmls = sorted(glob.glob(os.path.join(folder, "*.xml")))
    samples = []
    for xp in xmls:
        stem = os.path.splitext(os.path.basename(xp))[0]
        img_path = None
        for ext in img_exts:
            p = os.path.join(folder, stem + ext)
            if os.path.exists(p):
                img_path = p; break
        if img_path is None:
            continue
        W,H,gts = parse_voc(xp, label_filter)
        samples.append((img_path, gts, (W,H)))
    return samples

# -----------------------------
# IoU + NMS
# -----------------------------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
    inter_x1 = np.maximum(ax1[:,None], bx1[None,:])
    inter_y1 = np.maximum(ay1[:,None], by1[None,:])
    inter_x2 = np.minimum(ax2[:,None], bx2[None,:])
    inter_y2 = np.minimum(ay2[:,None], by2[None,:])
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h
    area_a  = (ax2 - ax1) * (ay2 - ay1)
    area_b  = (bx2 - bx1) * (by2 - by1)
    union   = area_a[:,None] + area_b[None,:] - inter
    return np.where(union > 0, inter / union, 0.0)

def nms(boxes, scores, iou_thresh):
    idxs = np.argsort(-scores)
    keep = []
    while idxs.size > 0:
        i = idxs[0]; keep.append(i)
        if idxs.size == 1: break
        rest = idxs[1:]
        ious = iou_xyxy(boxes[i:i+1], boxes[rest]).ravel()
        idxs = rest[ious < iou_thresh]
    return np.array(keep, dtype=np.int32)

# -----------------------------
# SavedModel detector (robust outputs + coords)
# -----------------------------
# class SavedModelDetector:
#     """
#     EfficientDet-Lite SavedModel (Model Maker). Typical outputs:
#       output_0: [B,100,4] boxes (ymin,xmin,ymax,xmax), normalized 0..1 OR pixels
#       output_2: [B,100]   scores
#       output_1: [B,100]   classes
#       output_3: [B]       num
#     """
#     def __init__(self, saved_model_dir):
#         assert os.path.isdir(saved_model_dir) and os.path.exists(os.path.join(saved_model_dir, "saved_model.pb")), \
#             f"Not a SavedModel dir: {saved_model_dir}"
#         self.model = tf.saved_model.load(saved_model_dir)
#         self.fn = self.model.signatures['serving_default']
#         self.outputs = self.fn.structured_outputs
#         self.keys = list(self.outputs.keys())
#         print("SavedModel output keys:", self.keys)
#
#         # Map boxes/scores keys
#         if "output_0" in self.keys and "output_2" in self.keys:
#             self.k_boxes, self.k_scores = "output_0", "output_2"
#         elif "detection_boxes" in self.keys and "detection_scores" in self.keys:
#             self.k_boxes, self.k_scores = "detection_boxes", "detection_scores"
#         else:
#             # Fallback: choose [*,*,4] as boxes and 2D float as scores
#             boxes_key, scores_key = None, None
#             for k,t in self.outputs.items():
#                 shp = t.shape
#                 if len(shp) == 3 and shp[-1] == 4:
#                     boxes_key = k
#                 elif len(shp) == 2 and t.dtype == tf.float32:
#                     scores_key = k
#             self.k_boxes = boxes_key or self.keys[0]
#             self.k_scores = scores_key or (self.keys[1] if len(self.keys)>1 else self.keys[0])
#             print(f"⚠️ Heuristic mapping -> boxes: {self.k_boxes}, scores: {self.k_scores}")
#
#         self._logged_once = False
#
#     @staticmethod
#     def _clip_xyxy(xyxy, W, H):
#         x1 = np.clip(xyxy[:,0], 0, W-1)
#         y1 = np.clip(xyxy[:,1], 0, H-1)
#         x2 = np.clip(xyxy[:,2], 0, W-1)
#         y2 = np.clip(xyxy[:,3], 0, H-1)
#         # ensure proper ordering
#         x1, x2 = np.minimum(x1, x2), np.maximum(x1, x2)
#         y1, y2 = np.minimum(y1, y2), np.maximum(y1, y2)
#         return np.stack([x1,y1,x2,y2], axis=1)
#
#     def infer(self, img_bgr):
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         tensor  = convert_to_tensor(img_rgb[None, ...], dtype=tf.uint8)
#         outs    = self.fn(tensor)
#
#         boxes  = outs[self.k_boxes][0].numpy()   # [N,4] (ymin,xmin,ymax,xmax), norm or pixel
#         scores = outs[self.k_scores][0].numpy()  # [N]
#
#         H0, W0 = img_bgr.shape[:2]
#
#         # Detect whether boxes are normalized or pixels
#         finite = np.isfinite(boxes).all()
#         if not finite:
#             boxes = np.nan_to_num(boxes, nan=0.0, posinf=W0, neginf=0.0)
#
#         max_val = boxes.max() if boxes.size else 0.0
#         min_val = boxes.min() if boxes.size else 0.0
#         normalized = (min_val >= -0.05) and (max_val <= 1.05)
#
#         # Convert [ymin,xmin,ymax,xmax] -> [x1,y1,x2,y2]
#         if normalized:
#             ymin, xmin, ymax, xmax = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
#             xyxy = np.stack([xmin*W0, ymin*H0, xmax*W0, ymax*H0], axis=1).astype(np.float32)
#         else:
#             # Assume pixel space already
#             ymin, xmin, ymax, xmax = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
#             xyxy = np.stack([xmin, ymin, xmax, ymax], axis=1).astype(np.float32)
#
#         xyxy = self._clip_xyxy(xyxy, W0, H0)
#
#         # One-time sanity log
#         if not self._logged_once and xyxy.size:
#             k = min(3, xyxy.shape[0])
#             print(f"[sanity] boxes min/max before scale: {min_val:.3f}/{max_val:.3f}  (normalized={normalized})")
#             print(f"[sanity] first {k} boxes (xyxy, clipped):\n{xyxy[:k]}")
#             print(f"[sanity] first {k} scores: {scores[:k]}")
#             self._logged_once = True
#
#         return xyxy, scores

class SavedModelDetector:
    """
    EfficientDet-Lite SavedModel (Model Maker). Typical outputs:
      output_0: [B,100,4] boxes (ymin,xmin,ymax,xmax), normalized 0..1 OR pixels
      output_1: [B,100]   classes  (float32 but integer-like values)
      output_2: [B,100]   scores   (0..1 probabilities)
      output_3: [B]       num_detections
    """
    def __init__(self, saved_model_dir):
        assert os.path.isdir(saved_model_dir) and os.path.exists(os.path.join(saved_model_dir, "saved_model.pb")), \
            f"Not a SavedModel dir: {saved_model_dir}"
        self.model = tf.saved_model.load(saved_model_dir)
        self.fn = self.model.signatures['serving_default']
        self.outputs = self.fn.structured_outputs
        self.keys = list(self.outputs.keys())
        print("SavedModel output keys:", self.keys)

        # Always boxes from output_0 (or fallback)
        self.k_boxes = "output_0" if "output_0" in self.keys else None
        if self.k_boxes is None:
            # Fallback: find [*,*,4]
            for k, t in self.outputs.items():
                shp = t.shape
                if len(shp) == 3 and shp[-1] == 4:
                    self.k_boxes = k; break
        assert self.k_boxes is not None, "Could not locate boxes output."

        # We'll decide scores/classes at runtime from output_1/output_2
        self.k_scores = None
        self.k_classes = None
        self.k_count = "output_3" if "output_3" in self.keys else None

        self._logged_once = False

    @staticmethod
    def _clip_xyxy(xyxy, W, H):
        x1 = np.clip(xyxy[:,0], 0, W-1); y1 = np.clip(xyxy[:,1], 0, H-1)
        x2 = np.clip(xyxy[:,2], 0, W-1); y2 = np.clip(xyxy[:,3], 0, H-1)
        x1, x2 = np.minimum(x1,x2), np.maximum(x1,x2)
        y1, y2 = np.minimum(y1,y2), np.maximum(y1,y2)
        return np.stack([x1,y1,x2,y2], axis=1)

    @staticmethod
    def _looks_like_scores(v):
        if v.size == 0 or not np.isfinite(v).all(): return False
        vmin, vmax, vstd = float(v.min()), float(v.max()), float(v.std())
        # scores sit in [0,1], with some spread
        return (vmin >= -1e-3) and (vmax <= 1.0 + 1e-3) and (vstd > 1e-3)

    def infer(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor  = tf.convert_to_tensor(img_rgb[None, ...], dtype=tf.uint8)
        outs    = self.fn(tensor)

        boxes  = outs[self.k_boxes][0].numpy()    # [N,4], norm or pixel
        # decide scores/classes dynamically
        cand1 = outs.get("output_1")
        cand2 = outs.get("output_2")
        if cand1 is None or cand2 is None:
            # OD API names?
            cand1 = outs.get("detection_scores")
            cand2 = outs.get("detection_classes")

        v1 = cand1[0].numpy() if cand1 is not None else np.array([])
        v2 = cand2[0].numpy() if cand2 is not None else np.array([])

        if self._looks_like_scores(v1) and not self._looks_like_scores(v2):
            scores, classes = v1, v2
            self.k_scores, self.k_classes = "output_1", "output_2"
        elif self._looks_like_scores(v2) and not self._looks_like_scores(v1):
            scores, classes = v2, v1
            self.k_scores, self.k_classes = "output_2", "output_1"
        else:
            # fallback: pick the one with more variance in [0,1]
            cands = []
            for name, v in (("output_1", v1), ("output_2", v2)):
                if v.size and np.isfinite(v).all():
                    cands.append((name, float(np.clip(v,0,1).std())))
            cands.sort(key=lambda x: x[1], reverse=True)
            if cands:
                self.k_scores = cands[0][0]
                self.k_classes = "output_2" if self.k_scores == "output_1" else "output_1"
                scores = outs[self.k_scores][0].numpy()
            else:
                raise RuntimeError("Could not identify scores among outputs.")

        # Use num_detections if present
        if self.k_count and self.k_count in outs:
            num = int(outs[self.k_count][0].numpy())
            boxes  = boxes[:num]
            scores = scores[:num]

        H0, W0 = img_bgr.shape[:2]

        # normalize vs pixel
        finite = np.isfinite(boxes).all()
        if not finite:
            boxes = np.nan_to_num(boxes, nan=0.0, posinf=W0, neginf=0.0)
        bmin, bmax = float(boxes.min()) if boxes.size else 0.0, float(boxes.max()) if boxes.size else 0.0
        normalized = (bmin >= -0.05) and (bmax <= 1.05)

        ymin, xmin, ymax, xmax = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        if normalized:
            xyxy = np.stack([xmin*W0, ymin*H0, xmax*W0, ymax*H0], axis=1).astype(np.float32)
        else:
            xyxy = np.stack([xmin, ymin, xmax, ymax], axis=1).astype(np.float32)

        xyxy = self._clip_xyxy(xyxy, W0, H0)

        if not self._logged_once and xyxy.size:
            k = min(3, xyxy.shape[0])
            print(f"[mapping] scores from: {self.k_scores} | classes from: {self.k_classes or 'N/A'}")
            print(f"[sanity] boxes min/max before scale: {bmin:.3f}/{bmax:.3f}  (normalized={normalized})")
            print(f"[sanity] first {k} boxes (xyxy, clipped):\n{xyxy[:k]}")
            smin, smax, smean = float(scores.min()), float(scores.max()), float(scores.mean())
            print(f"[sanity] scores stats: min={smin:.3f} max={smax:.3f} mean={smean:.3f}")
            self._logged_once = True

        return xyxy, scores

# -----------------------------
# Matching & metrics
# -----------------------------
def match_image(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    order = np.argsort(-pred_scores)
    used_gt = np.zeros((gt_boxes.shape[0],), dtype=bool)
    tp = fp = 0
    for i in order:
        if pred_scores[i] <= 0:  # skip zeros
            continue
        if gt_boxes.size == 0:
            fp += 1; continue
        ious = iou_xyxy(pred_boxes[i:i+1], gt_boxes).ravel()
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not used_gt[j]:
            tp += 1; used_gt[j] = True
        else:
            fp += 1
    fn = int(np.sum(~used_gt))
    return tp, fp, fn

def precision_recall_f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

# -----------------------------
# Sweep
# -----------------------------
def sweep(saved_model_dir, data_dir, sample_frac=1.0, iou_match=0.5,
          score_list=(0.55,0.60,0.65,0.70,0.75),
          nms_iou_list=(0.40,0.45,0.50,0.55,0.60),
          label_filter={"barbell"}):

    data = load_dataset(data_dir, label_filter=label_filter)
    if sample_frac < 1.0:
        data = data[:max(1, int(len(data)*sample_frac))]
    print(f"Loaded {len(data)} samples from {data_dir}")

    det = SavedModelDetector(saved_model_dir)
    results = []

    for thr in score_list:
        for nms_iou in nms_iou_list:
            TP=FP=FN=0
            for img_path, gts, (W,H) in tqdm(data, desc=f"thr={thr:.2f}, nms={nms_iou:.2f}", leave=False):
                img = cv2.imread(img_path)
                if img is None: continue

                boxes, scores = det.infer(img)

                # 1) filter by score
                m = scores >= thr
                boxes = boxes[m]; scores_sel = scores[m]
                if boxes.size == 0:
                    FN += len(gts); continue

                # 2) extra NMS
                keep = nms(boxes, scores_sel, nms_iou)
                boxes = boxes[keep]; scores_sel = scores_sel[keep]

                tp, fp, fn = match_image(boxes, scores_sel, gts, iou_thr=iou_match)
                TP += tp; FP += fp; FN += fn

            p, r, f1 = precision_recall_f1(TP, FP, FN)
            results.append((thr, nms_iou, p, r, f1, TP, FP, FN))

    results.sort(key=lambda x: (x[4], x[2]), reverse=True)
    return results

# -----------------------------
# CLI
# -----------------------------
def main():
    setup_gpu()
    ap = argparse.ArgumentParser(description="Sweep score threshold & NMS IoU on a SavedModel (GPU/CUDA).")
    ap.add_argument("--model", required=True, help="Path to SavedModel dir (contains saved_model.pb)")
    ap.add_argument("--data", required=True, help="Folder with images + VOC XML")
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--scores", type=str, default="0.55,0.6,0.65,0.7,0.75")
    ap.add_argument("--nms-ious", type=str, default="0.40,0.45,0.50,0.55,0.60")
    ap.add_argument("--label", type=str, default="barbell")
    args = ap.parse_args()

    assert os.path.isdir(args.model) and os.path.exists(os.path.join(args.model, "saved_model.pb")), \
        f"--model must be a SavedModel directory (got: {args.model})"

    score_list  = [float(x) for x in args.scores.split(",")]
    nms_iou_list= [float(x) for x in args.nms_ious.split(",")]
    label_filter= {args.label} if args.label else None

    results = sweep(args.model, args.data, args.sample_frac, args.iou_match,
                    score_list, nms_iou_list, label_filter)

    print("\n=== Top results (F1 @ IoU={:.2f}) ===".format(args.iou_match))
    print(" score_thr | nms_iou | precision | recall |   F1   |  TP   FP   FN")
    for (thr, nms_iou, p, r, f1, TP, FP, FN) in results[:10]:
        print(f"   {thr:6.2f} |  {nms_iou:6.2f} |   {p:7.3f} | {r:6.3f} | {f1:6.3f} | {TP:4d} {FP:4d} {FN:4d}")

    best = results[0]
    print("\nBest pair -> score_thresh: {:.2f}, extra_nms_iou: {:.2f} (F1={:.3f}, P={:.3f}, R={:.3f})"
          .format(best[0], best[1], best[4], best[2], best[3]))

if __name__ == "__main__":
    main()


