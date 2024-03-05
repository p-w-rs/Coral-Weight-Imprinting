# example_usage.py
from coral_machine import CoralMachine

model_path = "test_data/mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite"
label_path = "test_data/imagenet_labels.txt"
relabel_path = "test_data/imprinting_data_script/open_image_v4_subset"
image_path = "test_data/grace_hopper.bmp"

machine = CoralMachine(model_path, label_path, relabel_path)
print(machine.run_inference(image_path))

machine.run_imprint()
print(machine.run_inference(image_path))
