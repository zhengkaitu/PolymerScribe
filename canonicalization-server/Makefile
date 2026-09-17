export TAG ?= dev

# Canonicalizing a BigSMILES with nested stochastic objects can blow up the
# graph enumeration in canon_tools.py until the kernel OOM-kills the server.
# MEMORY_LIMIT bounds that blast radius and RESTART_POLICY brings the service
# back, so a single pathological structure cannot end a long batch run.
# Override either on the command line, e.g.
#     make start-canonicalization-service MEMORY_LIMIT=16g
export MEMORY_LIMIT ?= 8g
export RESTART_POLICY ?= on-failure:100

.PHONY: build-canonicalization-image start-canonicalization-service stop-canonicalization-service

build-canonicalization-image:
	docker build -t canonicalization-server:$(TAG) .

# Note: --restart and --rm are mutually exclusive, so this container is not
# auto-removed; stop-canonicalization-service removes it explicitly.
start-canonicalization-service:
	docker run -d \
		--name canonicalization-server \
		-p 3319:3319 \
		--memory=$(MEMORY_LIMIT) \
		--restart=$(RESTART_POLICY) \
		canonicalization-server:$(TAG)

# Leading '-' so a missing or already-stopped container is not an error,
# which keeps `make stop-... && make start-...` usable for recovery.
stop-canonicalization-service:
	-docker stop canonicalization-server
	-docker rm canonicalization-server
