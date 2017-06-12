all:
	python setup.py build
	sudo python setup.py install

test-pearson:
	python test/test_pearson.py

test-norms:
	python test/test_norms.py

test: test-pearson test-norms
