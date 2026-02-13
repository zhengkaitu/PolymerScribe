export TAG ?= dev

.PHONY: build start

build:
	docker build -t bigsmiles-server:$(TAG) .

start:
	docker run -d \
		--name bigsmiles-server \
		-p 3318:3318 \
		--rm bigsmiles-server:$(TAG)

stop:
	docker stop bigsmiles-server
