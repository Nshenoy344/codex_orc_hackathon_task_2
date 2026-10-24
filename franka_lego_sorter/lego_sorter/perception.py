"""Small deterministic perception pipeline for a top-down source-stack ROI.

Teach every physical shade under the actual camera/light settings. Never infer
shade labels from web RGB values or silently assign an unrecognized shade.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import cv2
import numpy as np


def roi_mask(shape, roi):
    h, w = shape[:2]
    if roi is None:
        raise ValueError('Teach/configure a source ROI [x,y,width,height] first')
    x, y, rw, rh = map(int, roi)
    if x < 0 or y < 0 or rw < 3 or rh < 3 or x+rw > w or y+rh > h:
        raise ValueError('ROI is outside image or too small')
    mask = np.zeros((h, w), np.uint8)
    mask[y:y+rh, x:x+rw] = 255
    return mask


def lab_feature(bgr, mask):
    pixels = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(float)
    selected = pixels[np.asarray(mask) > 0]
    if len(selected) < 20:
        raise ValueError('Too few color pixels; choose an interior brick patch')
    # Remove deepest shadows and bright stud reflections before robust median.
    lo, hi = np.percentile(selected[:, 0], [15, 85])
    interior = selected[(selected[:, 0] >= lo) & (selected[:, 0] <= hi)]
    return np.median(interior, axis=0)


@dataclass
class ColorResult:
    label: str | None
    distance: float
    margin: float
    feature_lab: list


def color_camera_signature(metadata):
    """Color-affecting camera settings, independent of depth encoding/profile.

    Older captures recorded requested RGB resolution at top level; newer captures
    may additionally record the actual native color profile. Normalize those
    representations to the same color dimensions/format while preserving the
    complete original metadata elsewhere as provenance.
    """
    signature={key:metadata[key] for key in ('serial','camera_name','enable_auto_exposure',
                'enable_auto_white_balance','exposure','white_balance') if key in metadata}
    profile=metadata.get('native_color_profile') or metadata.get('color_profile') or {}
    for key in ('width','height','fps'):
        value=profile.get(key,metadata.get(key))
        if value is not None:signature[key]=value
    # The legacy Frame contract is BGR8; actual native format is checked when
    # supplied rather than accepting an accidental RGB/BGR channel change.
    value=profile.get('format',metadata.get('color_format','bgr8'))
    signature['format']=str(value).lower().split('.')[-1]
    return signature


class ColorModel:
    def __init__(self, samples=None, max_distance=18.0, min_margin=5.0, metadata=None):
        self.samples = samples or {}
        self.metadata = metadata or {}
        self.max_distance, self.min_margin = float(max_distance), float(min_margin)

    @classmethod
    def load(cls, path, max_distance=18.0, min_margin=5.0):
        with open(path) as stream:
            data = yaml.safe_load(stream)
        if data.get('format') != 'opencv_lab_v1':
            raise ValueError('Unknown color model format')
        return cls(data['samples'], max_distance, min_margin, data.get('metadata', {}))

    def save(self, path):
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(yaml.safe_dump(dict(format='opencv_lab_v1', samples=self.samples, metadata=self.metadata), sort_keys=False))
        temp.replace(path)

    @property
    def labels(self):
        return set(self.samples)

    def validate_labels(self, expected):
        missing = set(map(str, expected)) - self.labels
        if missing:
            raise ValueError('Physical color teaching missing labels: ' + ', '.join(sorted(missing)))
        return True

    def validate_camera(self, metadata):
        expected = self.metadata.get('camera')
        if expected and color_camera_signature(metadata) != color_camera_signature(expected):
            raise ValueError('Camera/settings changed since physical color teaching; reteach shades or restore settings')
        return True

    def record_camera(self, metadata):
        self.validate_camera(metadata)
        self.metadata['camera'] = dict(metadata)

    def teach(self, label, bgr, mask_or_roi):
        if not str(label).strip():
            raise ValueError('Color label cannot be empty')
        mask = (roi_mask(bgr.shape, mask_or_roi) if np.asarray(mask_or_roi).ndim == 1
                else np.asarray(mask_or_roi))
        feature = lab_feature(bgr, mask)
        self.samples.setdefault(str(label), []).append(feature.tolist())
        return feature.tolist()

    def classify(self, bgr, mask):
        if len(self.samples) < 2:
            raise ValueError('Teach at least two shades before classification')
        feature = lab_feature(bgr, mask)
        distances = []
        for label, examples in self.samples.items():
            values = np.asarray(examples, dtype=float)
            if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
                raise ValueError(f'Invalid color samples for {label}')
            # Each lighting/view sample is a prototype. More than one teaching
            # example accommodates ordinary stud/face shading without merging classes.
            distance = float(np.min(np.linalg.norm(values - feature, axis=1)))
            distances.append((distance, label))
        distances.sort()
        best, second = distances[:2]
        margin = second[0] - best[0]
        accepted = best[0] <= self.max_distance and margin >= self.min_margin
        return ColorResult(best[1] if accepted else None, best[0], margin, feature.tolist())


@dataclass
class Detection:
    color_id: str | None
    point_camera_m: list
    long_axis_camera: list
    center_px: list
    yaw_rad: float
    confidence: float
    diagnostics: dict

    def to_dict(self):
        return asdict(self)


def detect_top_brick(frame, config, model):
    model.validate_camera(frame.metadata)
    bgr, depth = frame.color_bgr, frame.depth_m
    if bgr.shape[:2] != depth.shape:
        raise ValueError('Color and depth must be aligned and have identical dimensions')
    source = roi_mask(bgr.shape, config.get('roi'))
    valid = ((source > 0) & np.isfinite(depth) &
             (depth >= float(config.get('min_depth_m', .20))) &
             (depth <= float(config.get('max_depth_m', 1.2))))
    if np.count_nonzero(valid) < int(config.get('min_component_px', 50)):
        raise ValueError('No valid source depth')
    # Nearest connected depth surface; percentile prevents one flying pixel from
    # choosing the surface. ROI should enclose the input stack with little background.
    top = float(np.percentile(depth[valid], float(config.get('top_percentile', 5))))
    band = float(config.get('top_band_m', .006))
    top_mask = (valid & (depth <= top+band)).astype(np.uint8)*255
    top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(top_mask)
    candidates = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < int(config.get('min_component_px', 50)):
            continue
        component = (labels == index).astype(np.uint8)*255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < float(config.get('min_area_px', 50)):
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        interior = np.zeros(depth.shape, np.uint8)
        cv2.fillConvexPoly(interior, np.round(box).astype(int), 255)
        interior = cv2.erode(interior, np.ones((3, 3), np.uint8))
        interior[source == 0] = 0
        support = (interior > 0) & valid & (depth <= top+band)
        fraction = np.count_nonzero(support)/max(1, np.count_nonzero(interior))
        if fraction < float(config.get('min_depth_fraction', .55)):
            continue
        z = float(np.median(depth[support]))
        center = np.asarray(rect[0], float)
        p = frame.deproject(*center, z)
        edges = [(box[(i+1)%4]-box[i]) for i in range(2)]
        points = [frame.deproject(*(center+edge/2), z)-
                  frame.deproject(*(center-edge/2), z) for edge in edges]
        lengths = [float(np.linalg.norm(edge)) for edge in points]
        long_i = int(np.argmax(lengths))
        long, short = max(lengths), min(lengths)
        tolerance = float(config.get('size_tolerance_m', .008))
        if (abs(long-float(config.get('expected_length_m', .0318))) > tolerance or
                abs(short-float(config.get('expected_width_m', .0158))) > tolerance):
            continue
        # Restrict color to robust interior pixels on the top surface.
        color_mask = support.astype(np.uint8)*255
        result = model.classify(bgr, color_mask)
        axis = points[long_i] / max(lengths[long_i], 1e-9)
        diagnostics = dict(depth_m=z, depth_fraction=fraction, width_m=short,
                           length_m=long, area_px=area, color_distance=result.distance,
                           color_margin=result.margin, feature_lab=result.feature_lab,
                           box_px=box.tolist(), top_depth_m=top)
        confidence = float(np.clip(1-result.distance/max(model.max_distance, 1e-9), 0, 1))
        candidates.append((z, Detection(result.label, p.tolist(), axis.tolist(), center.tolist(),
                          float(np.arctan2(axis[1], axis[0])), confidence, diagnostics)))
    if not candidates:
        raise ValueError('No top component matches a 2x4 brick; adjust ROI/geometry or reobserve')
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and abs(candidates[0][0]-candidates[1][0]) < float(config.get('ambiguous_depth_m', .004)):
        raise ValueError('Multiple top bricks at similar height; narrow source ROI or separate them')
    return candidates[0][1]
