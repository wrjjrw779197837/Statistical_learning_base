import sys
import os
import shutil
import random
import time
from captcha.image import ImageCaptcha
CHAR_SET = [str(i) for i in range(0, 10)]
CHAR_SET_LEN = len(CHAR_SET)
CAPTCHA_LEN = 4

ABSPATH_DIR_NAME = os.path.abspath(os.path.dirname(__file__))
CAPTCHA_IMAGE_PATH = os.path.join(ABSPATH_DIR_NAME, 'image_train_data')
TEST_IMAGE_PATH = os.path.join(ABSPATH_DIR_NAME, 'image_test_data')
TEST_IMAGE_NUM = 50


def generate_train_image(char_set=CHAR_SET, char_set_len=CHAR_SET_LEN, captcha_image_path=CAPTCHA_IMAGE_PATH):
    cnt = 0
    num_of_image = 1
    image = ImageCaptcha()
    for i in range(CAPTCHA_LEN):
        num_of_image *= char_set_len

    if not os.path.exists(captcha_image_path):
        os.mkdir(captcha_image_path)

    for i in range(char_set_len):
        for j in range(char_set_len):
            for k in range(char_set_len):
                for l in range(char_set_len):
                    image_context = char_set[i] + char_set[j] + char_set[k] + char_set[l]
                    image.write(image_context, os.path.join(captcha_image_path, image_context+'.jpg'))

                    cnt += 1
                    sys.stdout.write('Created (%d %d)' % (cnt, num_of_image))
                    sys.stdout.flush()


def cut_test_set(captcha_image_path=CAPTCHA_IMAGE_PATH, test_image_path=TEST_IMAGE_PATH):
    file_name_list = []
    for file_path in os.listdir(captcha_image_path):
        train_image = file_path.split('/')[-1]
        file_name_list.append(train_image)
    random.seed(time.time())
    random.shuffle(file_name_list)

    if not os.path.exists(captcha_image_path):
        os.mkdir(captcha_image_path)
    if not os.path.exists(test_image_path):
        os.mkdir(test_image_path)

    for i in range(TEST_IMAGE_NUM):
        image_name = file_name_list[i]
        shutil.move(os.path.join(captcha_image_path, image_name),
                    os.path.join(test_image_path, image_name))


if __name__ == '__main__':
    generate_train_image()
    cut_test_set()
    sys.stdout.write('Finished!\n')
    sys.stdout.flush()
    print(CAPTCHA_IMAGE_PATH)

