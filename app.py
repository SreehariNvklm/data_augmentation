import streamlit as st
from src.components.image_augmentation import ImageAugmentation
import cv2
import numpy as np
from PIL import Image
import os
from zipfile import ZipFile
import io
import pathlib

st.set_page_config("Data Augmentator","🛠️")

st.title("Data Augmentation Web Environment")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def convert(img):
    image_augmentation = ImageAugmentation([img])
    final_image_set = image_augmentation.processed_images

    bgr_img_set = final_image_set[0]
    rgb_img_set = final_image_set[1]

    final_images = []

    final_images.append(bgr_img_set[0])
    final_images.append(bgr_img_set[1])
    final_images.append(bgr_img_set[2]/255)
    for i in range(10):
        final_images.append(bgr_img_set[3][i]*255)
        final_images.append(bgr_img_set[4][i])
        final_images.append(bgr_img_set[5][i]*255)
        final_images.append(bgr_img_set[6][i]*255)
    final_images.append(bgr_img_set[7])

    final_images.append(rgb_img_set[0])
    final_images.append(rgb_img_set[1])
    final_images.append(rgb_img_set[2]/255)
    for i in range(10):
        final_images.append(rgb_img_set[3][i]*255)
        final_images.append(rgb_img_set[4][i])
        final_images.append(rgb_img_set[5][i]*255)
        final_images.append(rgb_img_set[6][i]*255)
    final_images.append(rgb_img_set[7])

    for i in range(len(final_images)):
        string = "outputs/"+str(i+1)+".png"
        cv2.imwrite(string,np.array(final_images[i])*255)
    
    return True

def convert_toZip():
    buffer = io.BytesIO()
    folder = pathlib.Path("./outputs")
    
    with ZipFile(buffer, "w") as zip_object:
        for file in folder.iterdir():
            zip_object.write(file, arcname=file.name)
        # for folder_name,sub_folder,file_names in os.walk("./outputs"):
        #     for file_name in file_names:
        #         file_path = os.path.join(folder_name,file_name)
        #         zip_object.write(file_path)
    
    buffer.seek(0)
    return buffer

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Submit'):
        st.write("Image submitted successfully!")
        cv2.imwrite("image.png",np.array(image))
        image = cv2.imread("image.png")

        if convert(image):
            st.toast("Augmented images generated succesfully!", icon="🎉")
            zip_file = convert_toZip()
            st.download_button(
                label="Download zip File",
                data=zip_file,
                file_name="output.zip",
                mime="application/zip",
            )
        else:
            st.toast("Oops! An error happened")

else:
    st.write("Please upload an image to proceed.")