def TensorToImg(img, to_pil_img):
    img = (img + 1) / 2
    return to_pil_img(img)