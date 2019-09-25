import os
import re
import csv
import numpy as np
import wv_util as wv
from PIL import Image




# image directory (for bulk processing)
# path to xview_dir
chip_dir = "/Users/eddiebedada/datasets/xview/"
# path to train_dir
img_dir = os.path.join(chip_dir, 'train_images/')
# test with one image
src = os.path.join(chip_dir, 'train_images/24.tif')
# output_image_path
images_output = './ships/ships_train_images/'
# output path to xview bboxes for each image
xview_labels_output = './ships/ships_xview_labels/'
# output path to label converted to yolo format
yolo_labels_output = './ships/ships_yolo_labels/'
# bbox file extension
bboxes_text = '.txt'

label_id = 50

# json_file to read from
json_file = os.path.join(chip_dir, '50.geojson')


def image_name(src):
    """
    extracts the name of an image and its file extension

    :param src: single image name
    :return: file name
    """

    return src.split('/')[-1]


def get_labels_for_chip(src, json_file):
    """
    Get the bbox and class for a chip

    :param src: path to chip image
    :param json_file: path to json_file with labels
    :return: bbox coords and classes
    """

    coords, chips, classes = wv.get_labels(json_file)
    chip_name = image_name(src)
    coords = coords[chips == chip_name]
    classes = classes[chips == chip_name].astype(np.int64)

    return coords, classes


def get_names_for_classes(txt_file):
    """
    :param txt_file: path to class_labels file
    :return: dict of classes with labels
    """

    labels = {}

    with open(txt_file) as f:
        for row in csv.reader(f):
            labels[int(row[0].split(":")[0])] = row[0].split(":")[1]

        return labels


def chip_one_image(src, json_file):

    img = Image.open(src)
    arr = np.array(img)
    coords, classes = get_labels_for_chip(src, json_file)
    c_img, c_box, c_cls = wv.chip_image(img=arr, coords=coords, classes=classes, shape=(416, 416))
    return c_img, c_box, c_cls


def chip_and_save_image(src, json_file, image_path, file_extension='.jpg', prefered_label=label_id):

    c_img, c_box, c_cls = chip_one_image(src, json_file)
    selected_labels = {}
    selected_chips = []
    selected_bboxes = {}

    # extract labels
    for cls in c_cls:
        for value in c_cls[cls]:
            if value == prefered_label:
                selected_labels[cls] = c_cls[cls]

    chip_name = image_name(src)
    base_name = chip_name.split('.')[0]

    # extract bbox
    for label in selected_labels:
        for box_id in c_box:
            if box_id == label:
                selected_bboxes[box_id] = c_box[box_id]


    # write bboxes to text_file
    for key, val in selected_bboxes.items():

        labels_file_name = "{}{}_{}{}".format(images_output, base_name, key, bboxes_text)
        val_u = val.astype(np.uint64)

        for v in (key, val_u):
            final = str(v)[1:-1]
            #final = res[1:-1]
            f2 = final.replace("]", "")
            f3 = f2.replace("[", '')
            w = open(labels_file_name, "w+")

            w.write(f3)

    # save images containing the needed class
    for i, array in enumerate(c_img):
        for label in selected_labels:
            if label == i:
                expand_img = np.expand_dims(array, axis=0)
                selected_chips.append(expand_img)
                output_filename_n = "{}{}_{}{}".format(image_path, base_name, i, file_extension)
                print('saving', "{}_{}{}".format(base_name, i, file_extension))
                save_image = Image.fromarray(array)
                save_image.save(output_filename_n, "JPEG")

    print('task is completed!')


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return x, y, w, h


def convert_to_yolo(input_label_path, output_label_path):
    g = open("output.txt", "w")
    for file in os.listdir(input_label_path):

        if ".txt" in file:
            filename = file[:-4] + ".jpg"

            input_file = open(os.path.join(input_label_path + file))
            file = file[:-4] + '.txt'

            output_file = open(output_label_path + file, "w")

            file_path = images_output + filename

            g.write(file_path + "\n")
            for line in input_file.readlines():
                match = re.findall(r"(\d+)", line)

                if match:
                    xmin = float(match[0])
                    ymin = float(match[1])
                    xmax = float(match[2])
                    ymax = float(match[3])

                    b = (xmin, xmax, ymin, ymax)
                    im = Image.open(file_path)
                    size = im.size
                    bb = convert(size, b)

                    output_file.write("0" + " " + " ".join([str(a) for a in bb]) + "\n")



            output_file.close()
            input_file.close()
    g.close()
    print('done!')


def bulk_chip_process(img_dir, json_file):
    for r, d, f in os.walk(img_dir):
        for name in f:
            img = os.path.join(img_dir, name)

            chip_and_save_image(img, json_file, images_output)


#if __name__ == "__main___":
    # print(get_names_for_classes('xview_class_labels.txt'))
    # crds, cls = get_labels_for_chip(src, json_file)
#chip_one_image(src, json_file)
#chip_and_save_image(src, json_file, images_output)

#bulk_chip_process(img_dir, json_file)

convert_to_yolo(xview_labels_output, yolo_labels_output)