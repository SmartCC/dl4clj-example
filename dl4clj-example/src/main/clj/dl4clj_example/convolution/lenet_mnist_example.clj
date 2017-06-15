(ns dl4clj-example.convolution.lenet-mnist-example
  (:require [deeplearning4clj.eval.evaluation :as evalution]
            [deeplearning4clj.nn.conf.layers :as layers]
            [deeplearning4clj.nn.conf.multi-layer-configuration :as netconf]
            [deeplearning4clj.optimize.listeners :as l]
            [deeplearning4clj.utils.java.keyword-2-java :as k2j])
  (:import [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf Updater]
           [org.deeplearning4j.nn.conf.distribution UniformDistribution]
           [org.deeplearning4j.nn.conf.inputs InputType]
           [org.deeplearning4j.nn.conf.layers ConvolutionLayer$Builder DenseLayer$Builder OutputLayer$Builder SubsamplingLayer$Builder SubsamplingLayer$PoolingType]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.activations Activation]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.slf4j LoggerFactory])
  (:gen-class :name "dl4cljExample.convolution.lenetMnistExample"))

(defn -main
  [& opts]
  (let [n-channels 1
        output-num 10
        batch-size 64
        n-epochs 1
        iterations 1 
        seed 123
        rng-seed 123
        mnist-train (MnistDataSetIterator. batch-size true rng-seed)
        mnist-test (MnistDataSetIterator. batch-size false rng-seed)
        log (LoggerFactory/getLogger "dl4cljExample.convolution.MLPMnistTwoLayerExample")
        hidden-layer-0 (layers/create-layer
                        (ConvolutionLayer$Builder. (int-array [5 5]))
                        :n-in n-channels
                        :stride (int-array [1 1])
                        :n-out 20
                        :activation Activation/IDENTITY)
        hidden-layer-1 (layers/create-layer
                        (SubsamplingLayer$Builder. SubsamplingLayer$PoolingType/MAX)
                        :kernel-size (int-array [2 2])
                        :stride (int-array [2 2]))
        hidden-layer-2 (layers/create-layer
                        (ConvolutionLayer$Builder. (int-array [5 5]))
                        :stride (int-array [2 2])
                        :n-out 50
                        :activation Activation/IDENTITY)
        hidden-layer-3 (layers/create-layer
                        (SubsamplingLayer$Builder. SubsamplingLayer$PoolingType/MAX)
                        :kernel-size (int-array [2 2])
                        :stride (int-array [2 2]))
        hidden-layer-4 (layers/create-layer
                        (DenseLayer$Builder.)
                        :activation Activation/RELU
                        :n-out 500)
        output-layer (layers/create-layer
                      (OutputLayer$Builder. LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
                      :n-out output-num
                      :activation Activation/SOFTMAX)
        list-builder (netconf/create-list-builder
                      :learning-rate 0.01
                      :seed seed
                      :iterations iterations
                      :regularization true
                      :l2 0.005
                      :weight-init WeightInit/XAVIER
                      :optimization-algo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT
                      :updater Updater/NESTEROVS
                      :momentum 0.9)
        conf (netconf/multi-layer-configuration
              list-builder
              :layer [0 hidden-layer-0]
              :layer [1 hidden-layer-1]
              :layer [2 hidden-layer-2]
              :layer [3 hidden-layer-3]
              :layer [4 hidden-layer-4]
              :layer [5 output-layer]
              :set-input-type (InputType/convolutionalFlat 28 28 1)
              :pretrain true
              :pretrain false)
        listeners (list (ScoreIterationListener. 1))
        net (k2j/doto-keyword-2-java (MultiLayerNetwork. conf)
                                     :init nil
                                     :set-listeners  listeners)]
    (.info log "Train Model ...")
    (dotimes [i n-epochs]
      (.info log "*** Completed epoch {} ***" i)
      (.fit net mnist-train))
    (.info log "Evaluate model ...")
    (->> (doto (evalution/create-evalution output-num)
          (evalution/evalution-eval mnist-test net))
        (evalution/evaluation-stats)
        (.info log))))

