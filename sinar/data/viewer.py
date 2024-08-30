import cv2
from pathlib import Path

def draw_bounding_boxes(image, label_file):
    bounding_boxes = []
    with label_file.open("r") as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            bounding_boxes.append(((x1, y1), (x2, y2)))
    # draw bounding boxes
    for (x1, y1), (x2, y2) in bounding_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def main(args):
    dataset_dir = args.dataset_dir
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    for image_path in images_dir.glob("*.jpg"):
        image = cv2.imread(str(image_path))
        label_file = labels_dir / f"{image_path.stem}.txt"
        # read label file
        if label_file.exists():
            image = draw_bounding_boxes(image, label_file)
        # add text to image
        cv2.putText(image, image_path.stem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        txt = "press space to continue, q to quit"
        cv2.putText(image, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        # show image
        cv2.imshow("image", image)
        # wait for space key and continue next image
        if cv2.waitKey(0) == ord(" "):
            continue
        # break if q key is pressed
        if cv2.waitKey(0) == ord("q"):
            break

    cv2.destroyAllWindows()
