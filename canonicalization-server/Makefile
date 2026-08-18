export TAG ?= dev

.PHONY: build-canonicalization-image start-canonicalization-service stop-canonicalization-service

build-canonicalization-image:
	docker build -t canonicalization-server:$(TAG) .

start-canonicalization-service:
	docker run -d \
		--name canonicalization-server \
		-p 3319:3319 \
		--rm canonicalization-server:$(TAG)

stop-canonicalization-service:
	docker stop canonicalization-server
