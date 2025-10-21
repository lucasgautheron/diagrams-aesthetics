#!/usr/bin/env bash

aws s3 sync s3://lucasgautheron/diagrams-aesthetics static/tasks

aws s3 sync static/tasks/comparison-clusters-1 s3://cap-lucasgautheron/diagrams-aesthetics/tasks/comparison-clusters-1
aws s3 sync static/tasks/comparison-random-2 s3://cap-lucasgautheron/diagrams-aesthetics/tasks/comparison-random-2
aws s3 sync static/tasks/rating-1 s3://cap-lucasgautheron/diagrams-aesthetics/tasks/rating-1
aws s3 sync static/tasks/guess-image-title s3://cap-lucasgautheron/diagrams-aesthetics/tasks/guess-image-title
