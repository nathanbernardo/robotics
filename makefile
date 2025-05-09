.PHONY: run-kinect-app run-motor-app run-arduino-nano run-listener run-publisher run-image-generator

run-kinect-app:
	python packages/kinect_app.py

run-motor-app:
	python packages/motor_app.py

run-arduino-nano:
	python packages/arduino_nano_app.py

run-listener:
	python packages/listener.py

run-publisher:
	python packages/publisher.py

run-image-generator:
	python packages/libs/ob_utils/image_generator.py $(ARGS)

run-train-model:
	python packages/train_model.py

run-take-pictures:
	python packages/take_pictures.py
