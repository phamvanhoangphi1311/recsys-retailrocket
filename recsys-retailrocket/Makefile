# ============================================
# Makefile — shortcuts cho dev workflow
# Usage: make build-train, make test, make push-all
# ============================================

# Variables
AWS_ACCOUNT_ID ?= 123456789012
AWS_REGION ?= ap-southeast-1
ECR_REGISTRY = $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
VERSION ?= $(shell git rev-parse --short HEAD)

# Image names
TRAIN_IMAGE = $(ECR_REGISTRY)/recsys-train:$(VERSION)
SERVE_IMAGE = $(ECR_REGISTRY)/recsys-serve:$(VERSION)

# ============================================
# DEV COMMANDS
# ============================================

.PHONY: test
test:
	python -m pytest tests/ -v

.PHONY: lint
lint:
	python -m flake8 src/ api_gateway/ --max-line-length=120
	python -m mypy src/ --ignore-missing-imports

# ============================================
# DOCKER BUILD — local
# ============================================

.PHONY: build-train
build-train:
	docker build -f Dockerfile.train -t recsys-train:$(VERSION) .
	@echo "Built: recsys-train:$(VERSION)"

.PHONY: build-serve
build-serve:
	docker build -f api_gateway/Dockerfile -t recsys-serve:$(VERSION) .
	@echo "Built: recsys-serve:$(VERSION)"

.PHONY: build-all
build-all: build-train build-serve

# ============================================
# DOCKER TEST — run locally to verify
# ============================================

.PHONY: run-serve-local
run-serve-local:
	docker run -p 8000:8000 recsys-serve:$(VERSION)

# ============================================
# ECR PUSH
# ============================================

.PHONY: ecr-login
ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(ECR_REGISTRY)

.PHONY: push-train
push-train: ecr-login
	docker tag recsys-train:$(VERSION) $(TRAIN_IMAGE)
	docker push $(TRAIN_IMAGE)

.PHONY: push-serve
push-serve: ecr-login
	docker tag recsys-serve:$(VERSION) $(SERVE_IMAGE)
	docker push $(SERVE_IMAGE)

.PHONY: push-all
push-all: push-train push-serve