# This file contains all the utility functions used in the main program.
import cv2
import matplotlib.pyplot as plt
import glob 
import random
import os
import shutil

def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax

def plot_box(image, bboxes, labels):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # Denormalize the coordinates.
        xmin = int(x1*w)
        ymin = int(y1*h)
        xmax = int(x2*w)
        ymax = int(y2*h)

        thickness = max(2, int(w/275))
                
        cv2.rectangle(
            image, 
            (xmin, ymin), (xmax, ymax),
            color=(0, 0, 255),
            thickness=thickness
        )
    return image

# Function to plot images with the bounding boxes.
def plot_n_samples(image_paths, label_paths, num_samples):
    all_images = []
    all_images.extend(glob.glob(image_paths+'/*.jpg'))
    
    all_images.sort()

    num_images = len(all_images)
    print(num_images)
    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        j = random.randint(0,num_images-1)
        image_name = all_images[j]
        image_name = '.'.join(image_name.split(os.path.sep)[-1].split('.')[:-1])
        image = cv2.imread(all_images[j])
        with open(os.path.join(label_paths, image_name+'.txt'), 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                x_c, y_c, w, h = bbox_string.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels)
        plt.subplot(2, 2, i+1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis('off')
    plt.subplots_adjust(wspace=1)
    plt.tight_layout()
    plt.show()
    
def remove_folder_contents(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Delete file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Recursively delete subdirectory
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
            
import os
import random

def split_data_and_annotations(image_folder, annotation_folder, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, output_img_dir="K:\\ML\\Projects\\vista\\YYOOLLOO\\data\\images", label_dir = "K:\\ML\\Projects\\vista\\YYOOLLOO\\data\\labels"):
  # Check if output directory exists, create it if not
  if not os.path.exists(output_img_dir):
    os.makedirs(output_img_dir)
  if not os.path.exists(label_dir):
    os.makedirs(label_dir)

  # Subfolders for images
  train_dir = os.path.join(output_img_dir, "train")
  val_dir = os.path.join(output_img_dir, "val")
  test_dir = os.path.join(output_img_dir, "test")
  os.makedirs(train_dir)
  os.makedirs(val_dir)
  os.makedirs(test_dir)

  # Subfolders for annotations
  train_annot_dir = os.path.join(label_dir, "train")
  val_annot_dir = os.path.join(label_dir, "val")
  test_annot_dir = os.path.join(label_dir, "test")
  os.makedirs(train_annot_dir)
  os.makedirs(val_annot_dir)
  os.makedirs(test_annot_dir)

  # Get a list of all image and annotation file paths
  image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))]
  annotation_paths = [os.path.join(annotation_folder, f) for f in os.listdir(annotation_folder) if f.endswith(".txt")]  # Assuming .txt for annotations

  # Check if number of images and annotations match
  if len(image_paths) != len(annotation_paths):
    raise ValueError("Number of images and annotation files don't match!")

  # Shuffle the data together (image and corresponding annotation)
  combined_data = list(zip(image_paths, annotation_paths))
  random.shuffle(combined_data)

  # Split data based on ratios
  num_data = len(combined_data)
  train_split = int(num_data * train_ratio)
  val_split = train_split + int(num_data * val_ratio)

  train_data = combined_data[:train_split]
  val_data = combined_data[train_split:val_split]
  test_data = combined_data[val_split:]

  # Move images and annotations to their respective folders
  for image_path, annotation_path in train_data:
    os.rename(image_path, os.path.join(train_dir, os.path.basename(image_path)))
    os.rename(annotation_path, os.path.join(train_annot_dir, os.path.basename(annotation_path)))

  for image_path, annotation_path in val_data:
    os.rename(image_path, os.path.join(val_dir, os.path.basename(image_path)))
    os.rename(annotation_path, os.path.join(val_annot_dir, os.path.basename(annotation_path)))

  for image_path, annotation_path in test_data:
    os.rename(image_path, os.path.join(test_dir, os.path.basename(image_path)))
    os.rename(annotation_path, os.path.join(test_annot_dir, os.path.basename(annotation_path)))

  print(f"Data split and saved to {output_img_dir} directory:")
  print(f"- Train: {len(train_data)}")
