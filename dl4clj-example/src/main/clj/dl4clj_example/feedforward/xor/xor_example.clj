(ns dl4clj-example.feedforward.xor.xor-example
  (:use [deeplearning4clj.nd4clj.dataset]
        [deeplearning4clj.nn.conf layers multi-layer-configuration])
  (:require [deeplearning4clj.eval.evaluation :as evalution]
            [deeplearning4clj.optimize.listeners :as l])
  (:import [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf.distribution UniformDistribution]
           [org.deeplearning4j.nn.conf.layers DenseLayer OutputLayer]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.activations Activation]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction])
  (:gen-class :name "dl4cljExample.feedforward.xor.XorExample"))

(defn -main
  [& opts]
  (let [ds (dataset [[0 0] [1 0] [0 1] [1 1]]
                    [[1 0] [0 1] [0 1] [1 0]])
        hidden-layer (deflayer DenseLayer
                       :n-in 2
                       :n-out 4
                       :activation Activation/SIGMOID
                       :weight-init WeightInit/DISTRIBUTION
                       :dist [(UniformDistribution. 0 1)])
        output-layer (deflayer OutputLayer
                       :n-in 4
                       :n-out 2
                       :loss-function LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD
                       :activation Activation/SOFTMAX
                       :weight-init WeightInit/DISTRIBUTION
                       :dist [(UniformDistribution. 0 1)])
        list-builder (create-list-builder
                      :iterations 10000
                      :learning-rate 0.1
                      :seed 123
                      :use-drop-connect false
                      :optimization-algo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT
                      :bias-init 0
                      :mini-batch false)
        conf (multi-layer-configuration
              list-builder
              :layer [0 hidden-layer]
              :layer [1 output-layer]
              :pretrain false
              :backprop true)
        listeners (l/create-listeners (ScoreIterationListener. 100))
        net (def-multi-layer-network conf
              :init nil
              :set-listeners listeners
              :fit ds)]
    (-> (doto (evalution/create-evalution 2)
          (evalution/evalution-eval (.getLabels ds) (.getFeatureMatrix ds) net))
        (evalution/evaluation-stats)
        println)))
