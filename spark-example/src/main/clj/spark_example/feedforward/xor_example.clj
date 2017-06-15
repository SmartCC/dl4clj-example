(ns spark-example.feedforward.xor-example
  (:use [deeplearning4clj.nd4clj.dataset]
        [deeplearning4clj.nn.conf layers multi-layer-configuration]
        [deeplearning4clj.spark.spark])
  (:import  [org.deeplearning4j.nn.api OptimizationAlgorithm]
            [org.deeplearning4j.nn.conf.distribution UniformDistribution]
            [org.deeplearning4j.nn.conf.layers DenseLayer OutputLayer]
            [org.deeplearning4j.nn.weights WeightInit]
            [org.deeplearning4j.spark.impl.multilayer SparkDl4jMultiLayer]
            [org.deeplearning4j.spark.impl.paramavg ParameterAveragingTrainingMaster]
            [org.nd4j.linalg.activations Activation]
            [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction])
  (:gen-class :name "sparkExample.feedforward.XorExample"))


(defn create-multilayer-network-conf
  "配置神经网络"
  []
  (let [hidden-layer (deflayer DenseLayer
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
                      :mini-batch false)]
    (multi-layer-configuration list-builder
                               :layer [0 hidden-layer]
                               :layer [1 output-layer]
                               :pretrain false
                               :backprop true)))

(defn create-training-master
  "创建TraningMaster,dl4j通过trainingMaster运行在spark上"
  []
  (let [batch-size-per-worker 16] ;每批次处理的数据量
    (def-training-master
      ParameterAveragingTrainingMaster batch-size-per-worker
      :averagingFrequency 5
      :workerPrefetchNumBatches 2
      :batchSizePerWorker batch-size-per-worker)))

(defn -main
  [& opts]
  (let [sc (spark-conf "local[*]" "xor_example")
        ds (dataset [[0 0] [1 0] [0 1] [1 1]]
                    [[1 0] [0 1] [0 1] [1 0]])
        train-data (parallelize-dataset sc ds)
        conf (create-multilayer-network-conf)
        tm (create-training-master)
        spark-net (SparkDl4jMultiLayer. sc conf tm)]
    (.fit spark-net train-data)
    (-> (.evaluate spark-net train-data)
        (.stats)
        println)
    (.deleteTempFiles sc)))
