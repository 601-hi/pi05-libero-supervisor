#!/usr/bin/env python3
import cv2
import numpy as np

from export_wrist_multiscale_motion import flow_feature


def translated_square(dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    before = np.zeros((224, 224, 3), dtype=np.uint8)
    after = before.copy()
    cv2.rectangle(before, (72, 72), (152, 152), (255, 255, 255), -1)
    cv2.rectangle(after, (72 + dx, 72 + dy), (152 + dx, 152 + dy), (255, 255, 255), -1)
    return before, after


def main() -> None:
    right = flow_feature(*translated_square(6, 0))
    left = flow_feature(*translated_square(-6, 0))
    down = flow_feature(*translated_square(0, 6))
    assert right[:16].mean() > 0
    assert left[:16].mean() < 0
    assert down[16:32].mean() > 0
    assert np.allclose(flow_feature(*translated_square(0, 0)), 0, atol=1e-5)
    print('WRIST_MULTISCALE_SYNTHETIC_TEST_OK')


if __name__ == '__main__':
    main()
