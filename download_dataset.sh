#!/bin/bash
set -e
./scripts/train_pipeline.sh --lang te --no > ./docs/te_dataset_info.md
./scripts/train_pipeline.sh --lang bn --no > ./docs/bn_dataset_info.md
./scripts/train_pipeline.sh --lang mr --no > ./docs/mr_dataset_info.md
./scripts/train_pipeline.sh --lang gu --no > ./docs/gu_dataset_info.md
./scripts/train_pipeline.sh --lang kn --no > ./docs/kn_dataset_info.md
./scripts/train_pipeline.sh --lang ml --no > ./docs/ml_dataset_info.md
./scripts/train_pipeline.sh --lang pa --no > ./docs/pa_dataset_info.md
./scripts/train_pipeline.sh --lang or --no > ./docs/or_dataset_info.md
./scripts/train_pipeline.sh --lang as --no > ./docs/as_dataset_info.md
