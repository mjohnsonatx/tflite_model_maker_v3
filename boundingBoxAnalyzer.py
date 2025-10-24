import os, glob, random, xml.etree.ElementTree as ET
from statistics import mean, median
from math import isfinite

# =======================
# Config
# =======================
DATA_DIR = r"FINAL_BARBELL_DATA\train"   # <— your folder
SAMPLE_FRAC = 1                        # analyze ~50% of XMLs
# SAMPLE_K = 800                         # (optional) analyze exactly K files; comment out to use frac
SEED = 1337
LABEL_FILTER = {"barbell"}               # set to None to include all labels
REQUIRE_IMAGE_FILE = False               # True = only keep XMLs with a found image file

# Recognized image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =======================
# Helpers
# =======================
def pct(vals, p):
    if not vals: return None
    a = sorted(vals); i = (len(a) - 1) * (p / 100.0)
    lo, hi = int(i), min(int(i) + 1, len(a) - 1)
    if lo == hi: return a[lo]
    frac = i - lo
    return a[lo] * (1 - frac) + a[hi] * frac

def find_image_for_xml(xml_path):
    """Try to locate a matching image for an XML:
       1) use <filename> inside the XML (most robust),
       2) fall back to same stem with any known extension."""
    try:
        root = ET.parse(xml_path).getroot()
        fname_tag = root.findtext("filename")
        if fname_tag:
            img_path = os.path.join(os.path.dirname(xml_path), fname_tag)
            if os.path.exists(img_path):
                return img_path
    except Exception:
        pass
    # fallback by stem + any ext
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    folder = os.path.dirname(xml_path)
    for ext in IMG_EXTS:
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    return None

def voc_parse(xml_path):
    """Parse a Pascal VOC .xml -> (W, H, [(label, xmin, ymin, xmax, ymax), ...])"""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError("No <size> block")
    W = int(float(size.findtext("width")))
    H = int(float(size.findtext("height")))
    objs = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = int(float(bb.findtext("xmin")))
        xmax = int(float(bb.findtext("xmax")))
        ymin = int(float(bb.findtext("ymin")))
        ymax = int(float(bb.findtext("ymax")))
        # clamp
        xmin = max(0, min(xmin, W - 1)); xmax = max(0, min(xmax, W - 1))
        ymin = max(0, min(ymin, H - 1)); ymax = max(0, min(ymax, H - 1))
        if xmax <= xmin or ymax <= ymin:
            continue
        objs.append((name, xmin, ymin, xmax, ymax))
    return W, H, objs

# =======================
# Gather & sample XMLs
# =======================
all_xml = sorted(glob.glob(os.path.join(DATA_DIR, "*.xml")))
xmls = all_xml

if REQUIRE_IMAGE_FILE:
    kept = []
    for x in xmls:
        img = find_image_for_xml(x)
        if img is not None and os.path.splitext(img)[1].lower() in IMG_EXTS:
            kept.append(x)
    xmls = kept

if not xmls:
    print("No XMLs passed the filter. Check DATA_DIR and image extension rules.")
    print(f"Total XMLs found: {len(all_xml)} | REQUIRE_IMAGE_FILE={REQUIRE_IMAGE_FILE}")
    raise SystemExit

random.seed(SEED)
random.shuffle(xmls)

# Choose sample
SAMPLE_K = None  # comment this line and uncomment the config at top if you want a fixed K
if "SAMPLE_K" in globals() and isinstance(SAMPLE_K, int) and SAMPLE_K:
    xmls = xmls[:min(SAMPLE_K, len(xmls))]
else:
    xmls = xmls[:max(1, int(len(xmls) * SAMPLE_FRAC))]

# =======================
# Accumulate stats
# =======================
aspect = []        # width/height (signed: >1 means wider)
abs_aspect = []    # max(w,h)/min(w,h)
w_frac = []        # box width / image width
h_frac = []        # box height / image height
area_frac = []     # (w*h)/(W*H)
labels_seen = {}
total_boxes, images_with_boxes = 0, 0

for xml in xmls:
    try:
        W, H, objs = voc_parse(xml)
    except Exception as e:
        print(f"Skipping {xml}: {e}")
        continue
    if LABEL_FILTER:
        objs = [o for o in objs if o[0] in LABEL_FILTER]
    if not objs:
        continue
    images_with_boxes += 1
    for (name, xmin, ymin, xmax, ymax) in objs:
        labels_seen[name] = labels_seen.get(name, 0) + 1
        bw, bh = (xmax - xmin), (ymax - ymin)
        a = bw / bh
        aspect.append(a)
        abs_aspect.append(a if a >= 1.0 else 1.0/a)
        w_frac.append(bw / W); h_frac.append(bh / H)
        area_frac.append((bw * bh) / (W * H))
        total_boxes += 1

# =======================
# Report
# =======================
def fmt(x, dp=4):
    return "nan" if x is None or not isfinite(x) else f"{x:.{dp}f}"

print("=" * 80)
print("SAMPLE SUMMARY")
print("=" * 80)
print(f"Folder:                 {DATA_DIR}")
print(f"XML files total:        {len(all_xml)}")
print(f"Analyzed sample:        {len(xmls)} (~{int(SAMPLE_FRAC*100)}% if frac used)")
print(f"Images with boxes:      {images_with_boxes}")
print(f"Total boxes counted:    {total_boxes}")
print(f"Label filter:           {LABEL_FILTER if LABEL_FILTER else '(none)'}")
print(f"Labels seen:            {labels_seen if labels_seen else '(none)'}\n")

print("-" * 80)
print("ASPECT RATIO (width / height)")
print("-" * 80)
print(f"Mean:    {fmt(mean(aspect) if aspect else None)}")
print(f"Median:  {fmt(median(aspect) if aspect else None)}")
print(f"P10:     {fmt(pct(aspect, 10))}   P90: {fmt(pct(aspect, 90))}\n")

print("ABSOLUTE ASPECT (max(w,h)/min(w,h))")
print(f"Mean:    {fmt(mean(abs_aspect) if abs_aspect else None)}")
print(f"Median:  {fmt(median(abs_aspect) if abs_aspect else None)}")
print(f"P10:     {fmt(pct(abs_aspect, 10))}   P90: {fmt(pct(abs_aspect, 90))}\n")

print("-" * 80)
print("SIZE (normalized by image)")
print("-" * 80)
print(f"Width frac mean:        {fmt(mean(w_frac) if w_frac else None)}")
print(f"Height frac mean:       {fmt(mean(h_frac) if h_frac else None)}")
print(f"Area frac mean:         {fmt(mean(area_frac) if area_frac else None)}")
print(f"Area frac P10/P50/P90:  {fmt(pct(area_frac,10))} / {fmt(pct(area_frac,50))} / {fmt(pct(area_frac,90))}")
print("=" * 80)

# Simple suggestion for 3 ratios from your data (25/50/75th percentiles)
if aspect:
    p25, p50, p75 = pct(aspect, 25), pct(aspect, 50), pct(aspect, 75)
    suggestion = [round(p25, 2), round(p50, 2), round(p75, 2)]
    print("Suggested aspect_ratios (25/50/75th percentiles):", suggestion)
