from litecv import new_image


def main():
    img = new_image(480, 360, color='lightgray')
    img.draw_text('Basic Image Demo', (30, 30), color='black', size=30)
    img.draw_circle((240, 180), 70, color='red', fill='yellow')
    img.draw_rectangle((50, 240), (430, 310), color='blue', width=4, fill='lightblue')
    img.draw_line((50, 50), (430, 310), color='green', width=3)
    img.save('basic_image_output.jpg')
    print('Saved basic_image_output.jpg')


if __name__ == '__main__':
    main()
