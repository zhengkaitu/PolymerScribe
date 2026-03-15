export TAG ?= dev

.PHONY: \
	build-polymerscribe-image \
	download-polymerscribe-model \
	start-polymerscribe-service \
	stop-polymerscribe-service

build-polymerscribe-image:
	docker build -t polymerscribe-server:$(TAG) .

download-polymerscribe-model:
	sh scripts/download_polymerscribe_model.sh

start-polymerscribe-service:
	docker run -d --rm \
		--name polymerscribe-server \
		--network=host \
		-t polymerscribe-server:$(TAG)

stop-polymerscribe-service:
	docker stop polymerscribe-server
