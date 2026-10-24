"""Physical shade teaching and camera-only perception diagnostics."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from .camera import RealSenseCamera
from .perception import ColorModel, detect_top_brick, roi_mask


def save_snapshot(frame, output):
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    npz = output if output.suffix.lower() == '.npz' else output.with_suffix('.npz')
    image = output.with_suffix('.png') if output.suffix.lower() in ('.npz', '') else output
    if not cv2.imwrite(str(image), frame.color_bgr):
        raise RuntimeError(f'Could not write {image}')
    np.savez_compressed(npz, color_bgr=frame.color_bgr, depth_m=frame.depth_m,
                        intrinsics_json=json.dumps(frame.intrinsics),
                        metadata_json=json.dumps(frame.metadata),
                        captured_monotonic=frame.captured_monotonic)
    print(json.dumps(dict(snapshot=str(npz), color_image=str(image))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['teach', 'detect', 'preview', 'select-roi', 'snapshot'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', help='Override color model path')
    parser.add_argument('--label', help='Physical shade ID to teach, e.g. 300121')
    parser.add_argument('--roi', nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'))
    parser.add_argument('--output', help='Snapshot/annotated image path; snapshot also saves aligned-depth NPZ')
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    colors = config.get('colors', {})
    model_path = Path(args.model or config.get('color_model_file') or
                      colors.get('model_file', 'data/colors.yaml')).expanduser()
    # Shared convention: paths in config/demo.yaml are relative to repository root.
    if not model_path.is_absolute():
        root = config_path.parent.parent if config_path.parent.name == 'config' else config_path.parent
        model_path = root / model_path
    with RealSenseCamera(config.get('camera', {})) as camera:
        if args.action == 'preview':
            print('Preview only; Q/Escape exits. Camera must be stationary for teach/detect.', flush=True)
            while True:
                frame = camera.capture()
                cv2.imshow('RealSense color preview', frame.color_bgr)
                if cv2.waitKey(1) & 0xff in (27, ord('q')):
                    break
                if cv2.getWindowProperty('RealSense color preview', cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyAllWindows()
            return
        frame = camera.capture_burst()
        if args.action == 'snapshot':
            if not args.output:
                parser.error('snapshot requires --output, e.g. data/source.npz')
            save_snapshot(frame, args.output)
            return
        if args.action == 'select-roi':
            print('Draw a tight source-stack ROI. Enter accepts, Escape cancels.', flush=True)
            roi = args.roi or list(map(int, cv2.selectROI('Input stack ROI', frame.color_bgr,
                                                        showCrosshair=True, fromCenter=False)))
            cv2.destroyAllWindows()
            roi_mask(frame.color_bgr.shape, roi)  # validates cancel/out-of-bounds
            config.setdefault('perception', {})['roi'] = list(roi)
            temp = config_path.with_suffix(config_path.suffix+'.tmp')
            temp.write_text(yaml.safe_dump(config, sort_keys=False))
            temp.replace(config_path)
            if args.output:
                save_snapshot(frame, args.output)
            print(json.dumps(dict(config=str(config_path), source_roi=roi)))
            return
        model = (ColorModel.load(model_path, colors.get('max_distance', 18), colors.get('min_margin', 5))
                 if model_path.exists() else ColorModel(max_distance=colors.get('max_distance', 18),
                                                       min_margin=colors.get('min_margin', 5)))
        if args.output:
            save_snapshot(frame, args.output)
        if args.action == 'teach':
            if not args.label:
                parser.error('teach requires --label')
            expected = set(map(str, config.get('job', {}).get('good_colors', []) +
                                      config.get('job', {}).get('bad_colors', [])))
            if expected and args.label not in expected:
                parser.error('label must match a configured good_colors or bad_colors ID')
            print('Select an interior patch on one physical brick, avoiding edges/background. Enter accepts.', flush=True)
            roi = args.roi or list(map(int, cv2.selectROI('Teach shade: '+args.label, frame.color_bgr,
                                                        showCrosshair=True, fromCenter=False)))
            cv2.destroyAllWindows()
            model.record_camera(frame.metadata)
            feature = model.teach(args.label, frame.color_bgr, roi)
            model.save(model_path)
            print(json.dumps(dict(model=str(model_path), label=args.label, feature_lab=feature,
                                  example_count=len(model.samples[args.label]),
                                  taught_labels=sorted(model.labels),
                                  missing_labels=sorted(expected-model.labels))))
        else:
            job = config.get('job', {})
            model.validate_labels(job.get('good_colors', []) + job.get('bad_colors', []))
            perception = dict(config.get('perception', {}))
            if args.roi:
                perception['roi'] = args.roi
            detection = detect_top_brick(frame, perception, model)
            print(json.dumps(detection.to_dict(), indent=2))
            if args.output:
                annotated = frame.color_bgr.copy()
                cv2.polylines(annotated, [np.round(detection.diagnostics['box_px']).astype(int)],
                              True, (0, 255, 0), 2)
                cv2.putText(annotated, str(detection.color_id or 'UNKNOWN'),
                            tuple(map(int, detection.center_px)), cv2.FONT_HERSHEY_SIMPLEX,
                            .7, (0, 255, 0), 2)
                output = Path(args.output)
                cv2.imwrite(str(output.with_name(output.stem+'_annotated.png')), annotated)


if __name__ == '__main__':
    main()
