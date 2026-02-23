export TAG ?= dev

.PHONY: build-bigsmiles-image start-bigsmiles-service stop-bigsmiles-service

build-bigsmiles-image:
	docker build -t bigsmiles-server:$(TAG) .

start-bigsmiles-service:
	docker run -d \
		--name bigsmiles-server \
		-p 3318:3318 \
		--rm bigsmiles-server:$(TAG)

stop-bigsmiles-service:
	docker stop bigsmiles-server
