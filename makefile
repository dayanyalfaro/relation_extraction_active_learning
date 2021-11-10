build:
	docker build -t relation-al .

experiment1:
	docker run -v `pwd`:/home/coder/src -d relation-al bash -c /home/coder/src/experiment1.sh
