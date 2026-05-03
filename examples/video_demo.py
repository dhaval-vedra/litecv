from litecv import VideoProcessor


def main():
    processor = VideoProcessor()
    processor.open('example_video.mp4')
    for i in range(3):
        frame = processor.get_frame(i)
        frame.save(f'video_demo_frame_{i}.jpg')
        print(f'Saved video_demo_frame_{i}.jpg')

    processor.close()
    print('Video demo completed. Note: this is a placeholder video implementation.')


if __name__ == '__main__':
    main()
