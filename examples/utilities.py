from litecv import new_image, open_image, concatenate, blend_images


def main():
    img1 = new_image(240, 240, color='lightblue')
    img1.draw_text('Image 1', (20, 20), color='navy', size=20)
    img2 = new_image(240, 240, color='lightgreen')
    img2.draw_text('Image 2', (20, 20), color='darkgreen', size=20)

    concat = concatenate([img1.copy(), img2.copy()], direction='horizontal')
    concat.save('utilities_concat.jpg')
    print('Saved utilities_concat.jpg')

    blend = blend_images(img1, img2, alpha=0.4)
    blend.save('utilities_blend.jpg')
    print('Saved utilities_blend.jpg')

    loaded = open_image('utilities_concat.jpg')
    loaded.save('utilities_reload.jpg')
    print('Saved utilities_reload.jpg')


if __name__ == '__main__':
    main()
