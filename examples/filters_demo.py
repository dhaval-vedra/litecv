from litecv import new_image, FilterType, StreamingFilter


def main():
    img = new_image(320, 320, color='white')
    img.draw_text('Filters Demo', (40, 30), color='black', size=24)
    img.draw_circle((160, 180), 80, color='purple', fill='pink')
    img.draw_rectangle((60, 60), (260, 140), color='orange', width=5, fill='lightgreen')

    filters = [
        (FilterType.GRAYSCALE, 'grayscale'),
        (FilterType.EDGES, 'edges'),
        (FilterType.BLUR, 'blur'),
        (FilterType.SEPIA, 'sepia'),
        (FilterType.CARTOON, 'cartoon'),
        (FilterType.SKETCH, 'sketch'),
    ]

    for filter_type, name in filters:
        output = StreamingFilter(filter_type).apply(img.copy())
        output.save(f'filters_demo_{name}.jpg')
        print(f'Saved filters_demo_{name}.jpg')


if __name__ == '__main__':
    main()
