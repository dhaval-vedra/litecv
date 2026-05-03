from litecv import RealTimeCameraApp


def main():
    print('Starting LiteCV camera app. Use ESC to quit.')
    app = RealTimeCameraApp(resolution=(800, 600), camera_resolution=(640, 480))
    app.start()


if __name__ == '__main__':
    main()
