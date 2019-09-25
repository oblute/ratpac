import glob, os

# dir to images

script_dir = os.path.dirname(os.path.abspath(__file__))

for_aws_path = os.path.join(script_dir, 'ships/for_aws/')

# path starts listing from here
cargo_car_path = 'xview/Ships/yolo_ships_train_images/'

file_train = open('yolo_ships_train.txt', 'w')


counter = 1

for files in glob.iglob(os.path.join(for_aws_path, cargo_car_path, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(files))
    file_train.write(cargo_car_path + title + '.jpg' + "\n")
    counter = counter + 1
