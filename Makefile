all: python2

python2:
	python setup.py build
	sudo python setup.py install

python3:
	python3 setup.py build
	sudo python3 setup.py install

gfz:
	python setup.py build
	python setup.py install --user

clean:
	rm -rf build/
