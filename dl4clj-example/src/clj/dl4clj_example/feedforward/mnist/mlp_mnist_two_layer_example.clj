(ns dl4clj-example.feedforward.mnist.mlp-mnist-two-layer-example
  (:use [deeplearning4clj.nn.conf.layers]
        [deeplearning4clj.nn.conf.multi-layer-configuration])
  (:require [deeplearning4clj.eval.evaluation :as evalution]
            [deeplearning4clj.optimize.listeners :as l])
  (:import [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf Updater]
           [org.deeplearning4j.nn.conf.distribution UniformDistribution]
           [org.deeplearning4j.nn.conf.layers DenseLayer OutputLayer]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.activations Activation]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.slf4j LoggerFactory])
  (:gen-class :name "dl4cljExample.feedforward.minst.MLPMnistTwoLayerExample"))

(defn -main
  [& opts]
  (let [num-rows 28
        num-columns 28
        output-num 10
        batch-size 128
        rng-seed 123
        num-epochs 15
        rate 0.0015
        mnist-train (MnistDataSetIterator. batch-size true rng-seed)
        mnist-test (MnistDataSetIterator. batch-size false rng-seed)
        log (LoggerFactory/getLogger "dl4cljExample.feedforward.minstMLPMnistTwoLayerExample")
        hidden-layer-0 (deflayer DenseLayer
                         :n-in (* num-rows num-columns)
                         :n-out 500)
        hidden-layer-1 (deflayer DenseLayer
                         :n-in 500
                         :n-out 100)
        output-layer (deflayer OutputLayer
                       :loss-function LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD
                       :activation Activation/SOFTMAX
                       :n-in 100
                       :n-out output-num)
        list-builder (create-list-builder
                      :seed rng-seed
                      :optimization-algo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT
                      :iterations 1
                      :activation Activation/RELU
                      :weight-init WeightInit/XAVIER
                      :learning-rate rate
                      :updater Updater/NESTEROVS
                      :momentum 0.98
                      :regularization true
                      :l2  (* rate 0.005))
        conf (multi-layer-configuration
              list-builder
              :layer [0 hidden-layer-0]
              :layer [1 hidden-layer-1]
              :layer [2 output-layer]
              :pretrain false
              :backprop true)
        listeners (l/create-listeners (ScoreIterationListener. 5))
        net (def-multi-layer-network conf
              :init nil
              :set-listeners listeners)]
    (.info log "Train Model ... ")
    (dotimes [i num-epochs]
      (.info log (str "Epoch " i))
      (.fit net mnist-train))
    (->> (doto (evalution/create-evalution output-num)
           (evalution/evalution-eval mnist-test net))
         (evalution/evaluation-stats)
         (.info log))))
