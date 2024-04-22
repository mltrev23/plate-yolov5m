import yolov5
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load model
model = yolov5.load('keremberke/yolov5m-license-plate')
  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set image
#img = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
images = ['1.png', '1_rotated.png', '2.png', '3.png', '4.png']

# perform inference
for img in images:
    results = model(img, size=640)

    # inference with test time augmentation
    results = model(img, augment=True)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # show detection bounding boxes on image
    print("----------------------------")
    print(results)

    # save results into "results/" folder
    image = results.save()
    #image = mpimg.imread(path)  # Replace 'path_to_image.jpg' with the path to your image file
    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()
    print("----------------------------")

    print(predictions)
    print(boxes)
    print(scores)
    print(categories)