all:
	python setup.py build
	sudo python setup.py install

gfz:
	python setup.py build
	python setup.py install --user

