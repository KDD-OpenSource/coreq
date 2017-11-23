all:
	python2 setup.py build
	python3 setup.py build

install:
	sudo python2 setup.py install
	sudo python3 setup.py install

clean:
	python2 setup.py clean
	python3 setup.py clean

gfz:
	python setup.py build
	python setup.py install --user
