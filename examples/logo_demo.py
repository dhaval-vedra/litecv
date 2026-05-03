from litecv import get_logo_path, get_ascii_logo


def main():
    print("LiteCV Logo Examples")
    print("=" * 30)

    # Get logo paths
    svg_path = get_logo_path('svg')
    png_path = get_logo_path('png')
    ascii_path = get_logo_path('ascii')

    print(f"SVG Logo: {svg_path}")
    print(f"PNG Logo: {png_path}")
    print(f"ASCII Logo: {ascii_path}")
    print()

    # Display ASCII logo
    print("ASCII Logo:")
    print(get_ascii_logo())


if __name__ == '__main__':
    main()
