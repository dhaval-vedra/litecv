from litecv import new_image, FilterType, StreamingFilter, RealTimeCameraApp


def main():
    img = new_image(320, 240, color='skyblue')
    img.draw_text('LiteCV Example', (24, 24), color='black', size=24)
    img.draw_circle((160, 120), 50, color='red', fill='yellow')
    img.save('example_output.jpg')
    print('Saved example_output.jpg')

    gray = StreamingFilter(FilterType.GRAYSCALE).apply(img.copy())
    gray.save('example_output_gray.jpg')
    print('Saved example_output_gray.jpg')

    print('Starting realtime camera app...')
    app = RealTimeCameraApp(resolution=(640, 480), camera_resolution=(320, 240))
    app.start()


if __name__ == '__main__':
    main()
