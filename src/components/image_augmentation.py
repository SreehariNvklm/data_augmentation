import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import tensorflow as tf

class ImageAugmentation:
    def __init__(self,images):
        self.images = images
        self.processed_images = self.batching_images()

    def detect_edge_and_clip(self,image):
        edges = cv2.Canny(image,100,300)
        x,y,w,h = cv2.boundingRect(edges)
        cropped_img = image[y:y+h,x:x+w]
        converted_img = tf.image.convert_image_dtype(cropped_img, tf.float32)
        final_image = tf.image.resize(converted_img,(224,224))
        return final_image
    
    def make_rot_scale(self,image):
        data_rotate_flip = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
        ])
        rotated_images = []
        for i in range(10):
            augmented_image = data_rotate_flip(image/255)
            rotated_images.append(augmented_image)
        return rotated_images
    
    def color_grading(self,image):
        rgb_img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return (rgb_img,gray_img)
    
    def add_noise(self,image,noise_ratio=0.02):
        noise_img = image/255.0
        h,w,c = noise_img.shape
        noisy_pixels = int(h*w*noise_ratio)

        for _ in range(noisy_pixels):
            row, col = np.random.randint(0, h), np.random.randint(0, w)
            if np.random.rand() <= 0.5:
                noise_img[row][col] = [0,0,0]
            else:
                noise_img[row][col] = [255,255,255]
        return noise_img
    
    def add_blur_smoothening(self,image):
        smoothened_img = cv2.bilateralFilter(np.array(image),9,50,50)
        return smoothened_img
    
    def augment_img(self,image):
        nrm_image = self.detect_edge_and_clip(image)
        nrm_noise = self.add_noise(nrm_image.numpy()*255)
        rot_scaled_nrm = self.make_rot_scale(nrm_image)
        rot_scaled_nrm_noise = self.make_rot_scale(nrm_noise)
        rgb_nrm = cv2.cvtColor(nrm_image.numpy()*255,cv2.COLOR_BGR2RGB)
        nrm_blur = self.add_blur_smoothening(rgb_nrm)
        nrm_noise_blur = self.add_blur_smoothening(nrm_noise)
        nrm_scaled_rot_blur_imgs = []
        for i in range(10):
            nrm_scaled_rot_blur = self.add_blur_smoothening(rot_scaled_nrm[i]*255)
            nrm_scaled_rot_blur_imgs.append(nrm_scaled_rot_blur)
        nrm_scaled_rot_noise_blur_imgs = []
        for i in range(len(rot_scaled_nrm_noise)):
            nrm_scaled_rot_noise_blur = self.add_blur_smoothening(rot_scaled_nrm_noise[i])
            nrm_scaled_rot_noise_blur_imgs.append(nrm_scaled_rot_noise_blur)
        
        return nrm_image,nrm_noise,nrm_blur,rot_scaled_nrm,nrm_scaled_rot_blur_imgs,nrm_scaled_rot_noise_blur_imgs,rot_scaled_nrm_noise,nrm_noise_blur

    def batching_images(self):
        for img in self.images:
            normal_img,normal_noise,normal_blur,normal_rotated,normal_rotated_blur,normal_rotated_noise_blur,normal_rotated_noise,normal_noise_blur = self.augment_img(img)
            rgb_image, gray_image = self.color_grading(img)
            rgb_img,rgb_noise,rgb_blur,rgb_rotated,rgb_rotated_blur,rgb__rotated_noise_blur,rgb_rotated_noise,rgb_noise_blur = self.augment_img(rgb_image)
#             gray_img,gray_noise,gray_blur,gray_rotated,gray_rotated_blur,gray__rotated_noise_blur,gray_rotated_noise,gray_noise_blur = self.augment_img(gray_image)

            processed_img = []
            bgr = [normal_img,normal_noise,normal_blur,normal_rotated,normal_rotated_blur,normal_rotated_noise_blur,normal_rotated_noise,normal_noise_blur]
            rgb = [rgb_img,rgb_noise,rgb_blur,rgb_rotated,rgb_rotated_blur,rgb__rotated_noise_blur,rgb_rotated_noise,rgb_noise_blur]
#             gray = [gray_img,gray_noise,gray_blur,gray_rotated,gray_rotated_blur,gray__rotated_noise_blur,gray_rotated_noise,gray_noise_blur]

            processed_img.append(bgr)
            processed_img.append(rgb)
#             processed_img.append(gray)

            return processed_img

if __name__=="__main__":
    img = cv2.imread('profile.jpg')
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
    print("Augmented images obtained succesfully")

# class ImageAugmentation:
#     def __init__(self,images):
#         self.images = images
#         self.processed_images = self.batching_images()

#     def detect_edge_and_clip(self,image):
#         edges = cv2.Canny(image,100,300)
#         x,y,w,h = cv2.boundingRect(edges)
#         cropped_img = image[y:y+h,x:x+w]
#         converted_img = tf.image.convert_image_dtype(cropped_img, tf.float32)
#         final_image = tf.image.resize(converted_img,(224,224))
#         final_image = final_image.numpy()*255
#         return final_image
    
#     def make_rot_scale(self,image):
#         data_rotate_flip = tf.keras.Sequential([
#             tf.keras.layers.RandomFlip("horizontal_and_vertical"),
#             tf.keras.layers.RandomRotation(0.2),
#         ])
#         rotated_images = []
#         for i in range(10):
#             augmented_image = data_rotate_flip(image/255)
#             rotated_images.append(augmented_image)
#         return rotated_images
    
#     def color_grading(self,image):
#         rgb_img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#         gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#         return (rgb_img,gray_img)
    
#     def add_noise(self,image,noise_ratio=0.02):
#         noise_img = image/255
#         h,w,c = noise_img.shape
#         noisy_pixels = int(h*w*noise_ratio)
#         for _ in range(noisy_pixels):
#             row, col = np.random.randint(0, h), np.random.randint(0, w)
#             if np.random.rand() <= 0.5:
#                 noise_img[row][col] = [0,0,0]
#             else:
#                 noise_img[row][col] = [255,255,255]
#         return noise_img
    
#     def add_blur_smoothening(self,image):
#         smoothened_images = []
#         for i in range(10):
#             smoothened_img = cv2.bilateralFilter(np.array(image),9,50,50)
#             smoothened_images.append(smoothened_img)
#         return smoothened_images
    
#     def augment_img(self,image):
#         nrm_image = self.detect_edge_and_clip(image)
#         nrm_noise = self.add_noise(nrm_image)
#         rot_scaled_nrm = self.make_rot_scale(nrm_image)
#         rot_scaled_nrm_noise = self.make_rot_scale(nrm_noise)
#         nrm_blur = self.add_blur_smoothening(nrm_image)
#         nrm_noise_blur = self.add_blur_smoothening(nrm_noise)
#         nrm_scaled_rot_blur = self.add_blur_smoothening(rot_scaled_nrm)
#         nrm_scaled_rot_noise_blur = self.add_blur_smoothening(rot_scaled_nrm_noise)
        
#         return nrm_image,nrm_noise,nrm_blur,rot_scaled_nrm,nrm_scaled_rot_blur,nrm_scaled_rot_noise_blur,rot_scaled_nrm_noise,nrm_noise_blur

#     def batching_images(self):
#         for img in self.images:
#             normal_img,normal_noise,normal_blur,normal_rotated,normal_rotated_blur,normal_rotated_noise_blur,normal_rotated_noise,normal_noise_blur = self.augment_img(img)
#             rgb_image, gray_image = self.color_grading(img)
#             rgb_img,rgb_noise,rgb_blur,rgb_rotated,rgb_rotated_blur,rgb__rotated_noise_blur,rgb_rotated_noise,rgb_noise_blur = self.augment_img(rgb_image)
#             gray_img,gray_noise,gray_blur,gray_rotated,gray_rotated_blur,gray__rotated_noise_blur,gray_rotated_noise,gray_noise_blur = self.augment_img(gray_image)

#             processed_img = []
#             bgr = [normal_img,normal_noise,normal_blur,normal_rotated,normal_rotated_blur,normal_rotated_noise_blur,normal_rotated_noise,normal_noise_blur]
#             rgb = [rgb_img,rgb_noise,rgb_blur,rgb_rotated,rgb_rotated_blur,rgb__rotated_noise_blur,rgb_rotated_noise,rgb_noise_blur]
#             gray = [gray_img,gray_noise,gray_blur,gray_rotated,gray_rotated_blur,gray__rotated_noise_blur,gray_rotated_noise,gray_noise_blur]

#             processed_img.append(bgr)
#             processed_img.append(rgb)
#             processed_img.append(gray)

#             return processed_img
        
# if __name__=="__main__":
#     img = cv2.imread('messi 1.webp')
#     image_augmentation = ImageAugmentation([img])
#     final_image_set = image_augmentation.processed_images

#     bgr_img_set = final_image_set[0]
#     rgb_img_set = final_image_set[1]
#     gray_img_set = final_image_set[2]

#     for i in bgr_img_set:
#         plt.imshow(i)
