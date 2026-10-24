import argparse
import cv2
import yaml


def make_board(p):
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('OpenCV contrib (aruco) is required')
    name = p['dictionary']
    if not name.startswith('DICT_') or not hasattr(cv2.aruco, name):
        raise ValueError(f'Unknown ArUco dictionary: {name}')
    x, y = p['squares_x'], p['squares_y']
    square, marker = p['square_length_m'], p['marker_length_m']
    if x < 3 or y < 3 or not 0 < marker < square:
        raise ValueError('Need >=3 squares per axis and 0 < marker length < square length')
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    if hasattr(cv2.aruco, 'CharucoBoard_create'):
        board = cv2.aruco.CharucoBoard_create(x, y, square, marker, dictionary)
    else:
        board = cv2.aruco.CharucoBoard((x, y), square, marker, dictionary)
    if hasattr(board, 'setLegacyPattern'):
        board.setLegacyPattern(p.get('legacy_pattern', False))
    elif p.get('legacy_pattern', False):
        raise ValueError('legacy_pattern requires a newer OpenCV with setLegacyPattern')
    return board, dictionary


def detect(image, k, d, board, dictionary, min_corners):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, 'CharucoDetector'):
        params = cv2.aruco.CharucoParameters()
        params.cameraMatrix, params.distCoeffs = k, d
        corners, ids, _, _ = cv2.aruco.CharucoDetector(board, params).detectBoard(gray)
    else:
        markers, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        if marker_ids is None:
            raise ValueError('No ArUco markers detected')
        _, corners, ids = cv2.aruco.interpolateCornersCharuco(
            markers, marker_ids, gray, board, cameraMatrix=k, distCoeffs=d)
    if ids is None or len(ids) < max(4, min_corners):
        raise ValueError('Too few ChArUco corners')
    points = (board.getChessboardCorners() if hasattr(board, 'getChessboardCorners')
              else board.chessboardCorners)[ids.ravel()]
    import numpy as np
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError('ChArUco corners are collinear')
    ok, rv, tv = cv2.solvePnP(points, corners, k, d)
    if not ok or np.any((cv2.Rodrigues(rv)[0] @ points.T + tv)[2] <= 0):
        raise ValueError('Invalid board pose')
    projected, _ = cv2.projectPoints(points, rv, tv, k, d)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) -
                                        corners.reshape(-1, 2))**2, axis=1))))
    return rv, tv, rms, len(ids)


def main():
    parser = argparse.ArgumentParser(description='Generate board using calibration YAML')
    parser.add_argument('--params', required=True)
    parser.add_argument('--output', default='charuco.png')
    parser.add_argument('--pixels-per-square', type=int, default=200)
    args = parser.parse_args()
    with open(args.params) as stream:
        p = yaml.safe_load(stream)['handeye_calibrator']['ros__parameters']
    board, _ = make_board(p)
    size = (p['squares_x'] * args.pixels_per_square,
            p['squares_y'] * args.pixels_per_square)
    image = (board.generateImage(size) if hasattr(board, 'generateImage') else board.draw(size))
    if not cv2.imwrite(args.output, image):
        raise RuntimeError('Cannot write board image')
    print(f"Print board area at {p['squares_x'] * p['square_length_m'] * 1000:.1f} x "
          f"{p['squares_y'] * p['square_length_m'] * 1000:.1f} mm, without scaling; add white margin.")


if __name__ == '__main__':
    main()
