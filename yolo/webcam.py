from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (replace 'yolov8n.pt' with your model file if different)
model = YOLO('best (4).pt')

# Path to the input image
image_path = 'C:/Users/pramo/Downloads/yolo/images (1).jpeg'

# Perform prediction on the image
results = model.predict(source=image_path)

# Since `results` is a list, iterate through each result to display and save
for result in results:
    result.show()  # Display the image with detected objects

    # Optionally save the result
    result.save(save_dir='C:/Users/pramo/Downloads/yolo/output')  # Specify the directory to save the output image
