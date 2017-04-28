(ns dl4clj-example.feedforward.mnist.mlp-mnist-single-layer-example
   (:require [deeplearning4clj.eval.evaluation :as evalution]
            [deeplearning4clj.nd4clj.nd4clj :as nd4clj]
            [deeplearning4clj.nd4clj.dataset :as dataset]
            [deeplearning4clj.nn.conf.layers :as layers]
            [deeplearning4clj.nn.conf.multi-layer-configuration :as netconf]
            [deeplearning4clj.optimize.listeners :as l]
            [deeplearning4clj.utils.java.keyword-2-java :as k2j])
   (:import [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
            [org.deeplearning4j.nn.api OptimizationAlgorithm]
            [org.deeplearning4j.nn.conf Updater]
            [org.deeplearning4j.nn.conf.distribution UniformDistribution]
            [org.deeplearning4j.nn.conf.layers DenseLayer$Builder OutputLayer OutputLayer$Builder]
            [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
            [org.deeplearning4j.nn.weights WeightInit]
            [org.deeplearning4j.optimize.listeners ScoreIterationListener]
            [org.nd4j.linalg.activations Activation]
            [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
            [org.slf4j LoggerFactory])
   (:gen-class :name "dl4cljExample.feedforward.minst.MLPMnistSingleLayerExample"))

(defn -main
  [& opts]
  (let [num-rows 28
        num-columns 28
        output-num 10
        batch-size 128
        rng-seed 123
        num-epochs 15
        mnist-train (MnistDataSetIterator. batch-size true rng-seed)
        mnist-test (MnistDataSetIterator. batch-size false rng-seed)
        log (LoggerFactory/getLogger "dl4cljExample.feedforward.minst.MLPMnistSingleLayerExample")
        hidden-layer (layers/create-layer (DenseLayer$Builder.)
                                          :n-in (* num-rows num-columns)
                                          :n-out 1000
                                          :activation Activation/RELU
                                          :weight-init WeightInit/XAVIER)
        output-layer (layers/create-layer (OutputLayer$Builder.)
                                          :n-in 1000
                                          :n-out output-num
                                          :activation Activation/SOFTMAX
                                          :weight-init WeightInit/XAVIER)
        list-builder (netconf/create-list-builder
                      :seed rng-seed
                      :optimization-algo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT
                      :iterations 1
                      :learning-rate 0.006
                      :updater Updater/NESTEROVS
                      :momentum 0.9
                      :regularization true
                      :l1 1e-4)
       conf (netconf/multi-layer-configuration list-builder
              :layer [0 hidden-layer]
              :layer [1 output-layer]
              :pretrain false
              :backprop true)
        listeners (l/create-listeners (ScoreIterationListener. 100))
        net (k2j/doto-keyword-2-java (MultiLayerNetwork. conf)
                                     :init nil
                                     :set-listeners listeners)]
    (.info log "Train Model...")
    (dotimes [_ num-epochs]
      (.fit net mnist-train))
    (.info log "Evaluate Model...")
    (->> (doto (evalution/create-evalution output-num)
          (evalution/evalution-eval mnist-test net))
        (evalution/evaluation-stats)
        (.info log))))
