#!/bin/sh
./train.py --epochs 500 --lr 0.0005 --optimizer SGD --batch-size 512 --compress schedule_spo2.yaml --model ai85spo2net --dataset SPO2 --param-hist --device MAX78000 --regression --qat-policy qat_policy_spo2.yaml --print-freq 200 --validation-split 0 "$@"
