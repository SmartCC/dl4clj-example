(ns dl4clj-example.feedforward.xor.xor-example
  (:require [deeplearning4clj.eval.evaluation :as evalution]
            [deeplearning4clj.nd4clj.nd4clj :as nd4clj]
            [deeplearning4clj.nd4clj.dataset :as dataset]
            [deeplearning4clj.nn.conf.layers :as layers]
            [deeplearning4clj.nn.conf.multi-layer-configuration :as netconf]
            [deeplearning4clj.optimize.listeners :as l]
            [deeplearning4clj.utils.java.keyword-2-java :as k2j])
  (:import [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf.distribution UniformDistribution]
           [org.deeplearning4j.nn.conf.layers DenseLayer$Builder OutputLayer OutputLayer$Builder]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.activations Activation]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction])
  (:gen-class :name "dl4cljExample.feedforward.xor.XorExample"))

(defn -main
  [& opts]
  (let [ds (dataset/dataset [[0 0] [1 0] [0 1] [1 1]]
                            [[1 0] [0 1] [0 1] [1 0]])
        hidden-layer (layers/create-layer (DenseLayer$Builder.)
                                          :n-in 2
                                          :n-out 4
                                          :activation Activation/SIGMOID
                                          :weight-init WeightInit/DISTRIBUTION
                                          :dist [(UniformDistribution. 0 1)])
        output-layer (layers/create-layer (OutputLayer$Builder.)
                                          :n-in 4
                                          :n-out 2
                                          :loss-function LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD
                                          :activation Activation/SOFTMAX
                                          :weight-init WeightInit/DISTRIBUTION
                                          :dist [(UniformDistribution. 0 1)])
        list-builder (netconf/create-list-builder
              :iterations 1000
              :learning-rate 0.1
              :seed 123
              :use-drop-connect false
              :optimization-algo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT
              :bias-init 0
              :mini-batch false)
        conf (netconf/multi-layer-configuration list-builder
              :layer [0 hidden-layer]
              :layer [1 output-layer]
              :pretrain false
              :backprop true)
        listeners (l/create-listeners (ScoreIterationListener. 100))
        net (k2j/doto-keyword-2-java (MultiLayerNetwork. conf)
             :init nil
             :set-listeners listeners ;setListeners的参数是一个集合，原示例中是直接设置Listeners，此处设置不成功，需要添加到集合后才能设置
             :fit ds)]
    (-> (doto (evalution/create-evalution 2)
          (evalution/evalution-eval (.getLabels ds) (.getFeatureMatrix ds) net))
        (evalution/evaluation-stats)
        println)))
