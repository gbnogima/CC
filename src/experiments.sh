#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:.

PY='python3 main.py'
datapath='../datasets'

# run the lower-bound baseline Maximum Likelihood Prevalence Estimator
for dataset in kindle hp imdb ; do
  $PY --dataset $datapath/$dataset --method mlpe --learner none --error none --suffix "run0"
done

# runs the CC variants, with different learners, optimizing different evaluation metrics
for dataset in kindle hp imdb ; do
  for method in cc acc pcc pacc ; do
    for learner in svm lr mnb rf cnn ; do
      for error in none acce f1e mae ; do
        for run in {0..9} ; do
          $PY --dataset $datapath/$dataset --method $method --learner $learner --error $error --suffix "run"$run
        done
      done
    done
  done
done

# run the baselines
for dataset in kindle hp imdb ; do
  $PY --dataset $datapath/$dataset --method emq --learner lr --error mae --suffix "run0"
  $PY --dataset $datapath/$dataset --method svmkld --learner svmperf --error mae --suffix "run0"
  $PY --dataset $datapath/$dataset --method svmnkld --learner svmperf --error mae --suffix "run0"
  $PY --dataset $datapath/$dataset --method svmq --learner svmperf --error mae --suffix "run0"
  $PY --dataset $datapath/$dataset --method svmae --learner svmperf --error mae --suffix "run0"
  $PY --dataset $datapath/$dataset --method svmae --learner svmperf --error mrae --suffix "run0"
  $PY --dataset $datapath/$dataset --method svmrae --learner svmperf --error mae --suffix "run0"
  $PY --dataset $datapath/$dataset --method svmrae --learner svmperf --error mrae --suffix "run0"
  for run in {0..9} ; do
      $PY --dataset $datapath/$dataset --method hdy --learner lr --error mae --suffix "run"$run
      $PY --dataset $datapath/$dataset --method quanet --learner cnn --error mae --suffix "run"$run
  done
done

# generate the tables
$PY tables.py

