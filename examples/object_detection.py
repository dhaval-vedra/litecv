from litecv import open_image, ObjectDetector, new_image


def main():
    try:
        img = open_image('basic_image_output.jpg')
    except FileNotFoundError:
        img = new_image(320, 240, color='lightgray')
        img.draw_text('Sample Detection', (20, 20), color='black', size=20)
        img.draw_circle((160, 140), 60, color='blue', fill='lightblue')
        img.save('basic_image_output.jpg')
        print('Created basic_image_output.jpg for object detection sample.')

    detector = ObjectDetector()
    detections = detector.detect(img, confidence_threshold=0.5)

    for i, det in enumerate(detections, start=1):
        x, y, w, h = det['bbox']
        img.draw_rectangle((x, y), (x + w, y + h), color='red', width=3)
        img.draw_text(f"{det['label']} {det['confidence']:.2f}", (x, y - 20), color='yellow', size=14)
        print(f"Detection {i}: {det['label']} ({det['confidence']:.2f})")

    img.save('object_detection_output.jpg')
    print('Saved object_detection_output.jpg')


if __name__ == '__main__':
    main()
