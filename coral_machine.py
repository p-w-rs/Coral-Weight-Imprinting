# coral_machine.py
import os
import argparse
import numpy as np

from PIL import Image
from pycoral.adapters import common, classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file
from pycoral.learn.imprinting.engine import ImprintingEngine


class CoralMachine:
    def __init__(self, model_path, label_path, relabel_path):
        self.model_path = model_path
        self.label_path = label_path
        self.relabel_path = relabel_path
        self.last_results = None

        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.imprinting_engine = ImprintingEngine(model_path, keep_classes=True)
        self.labels = read_label_file(self.label_path)

    def run_inference(self, image_path):
        image = Image.open(image_path)
        _, scale = common.set_resized_input(
            self.interpreter,
            image.size,
            lambda size: image.resize(size, Image.Resampling.LANCZOS),
        )
        self.interpreter.invoke()
        classes = classify.get_classes(self.interpreter, top_k=5)
        self.last_results = [
            (self.labels.get(class_id.id, "Unknown"), class_id.score)
            for class_id in classes
        ]
        return self.last_results

    def _read_data(self, test_ratio):
        train_set = {}
        test_set = {}
        for category in os.listdir(self.relabel_path):
            category_dir = os.path.join(self.relabel_path, category)
            if os.path.isdir(category_dir):
                images = [
                    f
                    for f in os.listdir(category_dir)
                    if os.path.isfile(os.path.join(category_dir, f))
                ]
                if images:
                    k = max(int(test_ratio * len(images)), 0)
                    test_set[category] = images[:k]
                    train_set[category] = images[k:]
                    assert train_set[category], "No images to train [{}]".format(
                        category
                    )
        return train_set, test_set

    def _prepare_images(self, image_list, directory, shape):
        ret = []
        for filename in image_list:
            with Image.open(os.path.join(directory, filename)) as img:
                img = img.convert("RGB")
                img = img.resize(shape, Image.NEAREST)
                ret.append(np.asarray(img))
        return np.array(ret)

    def _update_labels_map(self, categories):
        existing_labels = {
            v: k for k, v in self.labels.items()
        }  # Reverse mapping for label to class ID
        labels_map = {}
        class_id_offset = (
            max(self.labels.keys(), default=-1) + 1
        )  # Start from the next available class ID
        for category in categories:
            if category in existing_labels:
                labels_map[category] = existing_labels[category]
            else:
                labels_map[category] = class_id_offset
                class_id_offset += 1
        return labels_map

    def _train_model(self, train_set, labels_map):
        extractor = make_interpreter(self.imprinting_engine.serialize_extractor_model())
        extractor.allocate_tensors()
        shape = common.input_size(extractor)

        for category, image_list in train_set.items():
            class_id = labels_map[category]
            prepared_images = self._prepare_images(
                image_list, os.path.join(self.relabel_path, category), shape
            )
            for tensor in prepared_images:
                common.set_input(extractor, tensor)
                extractor.invoke()
                embedding = classify.get_scores(extractor)
                self.imprinting_engine.train(embedding, class_id=class_id)
            self.labels[class_id] = category

        self.interpreter = make_interpreter(self.imprinting_engine.serialize_model())
        self.interpreter.allocate_tensors()

    def _save_retrained_model_and_labels(self):
        retrained_model_path = self.model_path.replace(".tflite", "_retrained.tflite")
        with open(retrained_model_path, "wb") as f:
            f.write(self.imprinting_engine.serialize_model())

        retrained_label_path = self.label_path.replace(".txt", "_retrained.txt")
        with open(retrained_label_path, "w") as f:
            for label_id, label in sorted(
                self.labels.items(), key=lambda item: item[0]
            ):
                f.write(f"{label_id} {label}\n")

        print(
            f"Retrained model saved to {retrained_model_path}. Updated labels are saved to {retrained_label_path}."
        )

    def run_imprint(self, test_ratio=0.0):
        train_set, _ = self._read_data(test_ratio)
        categories = list(train_set.keys())
        labels_map = self._update_labels_map(categories)
        self._train_model(train_set, labels_map)
        self._save_retrained_model_and_labels()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference and imprinting test on Coral device."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="test_data/mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite",
        help="Path to the model file.",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="test_data/imagenet_labels.txt",
        help="Path to the label file.",
    )
    parser.add_argument(
        "--relabel_path",
        type=str,
        default="test_data/imprinting_data_script/open_image_v4_subset",
        help="Path to the directory containing new data for imprinting.",
    )
    parser.add_argument(
        "--test_image_path",
        type=str,
        default="test_data/grace_hopper.bmp",
        help="Path to a test image.",
    )

    machine = CoralMachine(model_path, label_path, relabel_path)
    while True:
        print("Running inference 1000 times")
        for i in range(1000):
            machine.run_inference(test_image_path)
        print("Done")

        print("Running imprint 10 times")
        for i in range(2):
            machine.run_imprint()
        print("Done")
