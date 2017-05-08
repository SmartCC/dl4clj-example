(ns dl4clj-example.convolution.animals-classification
  (:require [deeplearning4clj.eval.evaluation :as evalution]
            [deeplearning4clj.nn.conf.layers :as layers]
            [deeplearning4clj.nn.conf.multi-layer-configuration :as netconf]
            [deeplearning4clj.optimize.listeners :as l]
            [deeplearning4clj.utils.java.keyword-2-java :as k2j])
  (:import [java.io File]
           [java.util ArrayList Random]
           [org.apache.commons.io FilenameUtils]
           [org.datavec.api.io.filters BalancedPathFilter]
           [org.datavec.api.io.labels ParentPathLabelGenerator]
           [org.datavec.api.split FileSplit InputSplit]
           [org.datavec.image.loader NativeImageLoader]
           [org.datavec.image.recordreader ImageRecordReader]
           [org.datavec.image.transform FlipImageTransform WarpImageTransform]
           [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator]
           [org.deeplearning4j.datasets.iterator MultipleEpochsIterator]
           [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf GradientNormalization LearningRatePolicy Updater]
           [org.deeplearning4j.nn.conf.distribution GaussianDistribution NormalDistribution]
           [org.deeplearning4j.nn.conf.inputs InputType InvalidInputTypeException]
           [org.deeplearning4j.nn.conf.layers ConvolutionLayer$Builder DenseLayer$Builder LocalResponseNormalization$Builder OutputLayer$Builder SubsamplingLayer$Builder SubsamplingLayer$PoolingType]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.activations Activation]
           [org.nd4j.linalg.dataset.api.preprocessor ImagePreProcessingScaler]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.slf4j LoggerFactory])
  (:gen-class :name "dl4cljExample.convolution.AnimalsClassification"))

(def height 1000)
(def width 1000)
(def channels 3)
(def num-examples 80)
(def num-labels 4)
(def batch-size 20)
(def seed 42)
(def rng (Random. seed))
(def listener-freq 1)
(def iterations 1)
(def epochs 50)
(def split-train-test (double-array [0.8 0.2])) ;FileSplit接收的是可变参数，实际是一个double类型的数组
(def n-cores 2)
(def save false)

(def model-type "LaNet") ;使用的模型

(defn conv-init
  "卷积层初始化"
  [name in out kernel stride pad bias]
  (let [[kernel stride pad] (map int-array [kernel stride pad])]
    (layers/create-layer (ConvolutionLayer$Builder. kernel stride pad)
                         :name name
                         :n-in in
                         :n-out out
                         :bias-init bias)))

(defn conv-3x3
  "创建3x3的卷积层"
  [name out bias]
  (let [[kernel stride pad] (map int-array [[3 3] [1 1] [1 1]])]
    (layers/create-layer (ConvolutionLayer$Builder. kernel stride pad)
                         :name name
                         :n-out out
                         :bias-init bias)))

(defn conv-5x5
  "创建5x5的卷积层"
  [name out stride pad bias]
  (let [[kernel stride pad] (map int-array [[5 5] stride pad])]
    (layers/create-layer (ConvolutionLayer$Builder. kernel stride pad)
                         :name name
                         :n-out out
                         :bias-init bias)))

(defn max-pool
  "池化层，使用最大池化法"
  [name kenel]
  (layers/create-layer (SubsamplingLayer$Builder. (int-array kenel) (int-array [2 2]))
                       :name name))

(defn fully-connected
  "全连接层"
  [name out bias drop-out dist]
  (layers/create-layer (DenseLayer$Builder.)
                       :name name
                       :n-out out
                       :bias-init bias
                       :drop-out drop-out
                       :dist dist))


(defn lanet-model
  "使用lanet模型"
  []
  (let [list-builder (netconf/create-list-builder
                      :seed seed
                      :iterations iterations
                      :regularization false
                      :l2 0.005
                      :activation Activation/RELU
                      :learning-rate 1e-4
                      :weight-init WeightInit/XAVIER
                      :optimization-algo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT
                      :updater Updater/RMSPROP
                      :momentum 0.9)
        hidden-layer-0 (conv-init "cnn1" channels 50 [5 5] [1 1] [0 0] 0)
        hidden-layer-1 (max-pool "maxpool1" [2 2])
        hidden-layer-2 (conv-5x5 "cnn2" 100 [5 5] [1 1] 0)
        hidden-layer-3 (max-pool "maxpool22" [2 2])
        hidden-layer-4 (layers/create-layer (DenseLayer$Builder.)
                                            :n-out 500)
        output-layer (layers/create-layer (OutputLayer$Builder. LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
                                          :n-out num-labels
                                          :activation Activation/SOFTMAX)
        conf (netconf/multi-layer-configuration
              list-builder
              :layer [0 hidden-layer-0]
              :layer [1 hidden-layer-1]
              :layer [2 hidden-layer-2]
              :layer [3 hidden-layer-3]
              :layer [4 hidden-layer-4]
              :layer [5 output-layer]
              :backprop true
              :pretrain false
              :set-input-type (InputType/convolutionalFlat height width channels))]
    (MultiLayerNetwork. conf)))

(defn alexnet-model
  "使用alexnet模型"
  []
  (let [non-zero-bias 1
        drop-out 0.5
        list-builder (netconf/create-list-builder
                      :seed seed
                      :weight-init WeightInit/DISTRIBUTION
                      :dist (NormalDistribution. 0.0 0.01)
                      :activation Activation/RELU
                      :updater Updater/NESTEROVS
                      :iterations iterations
                      :gradient-normalization GradientNormalization/RenormalizeL2PerLayer
                      :optimization-algo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT
                      :learning-rate 1e-2
                      :bias-learning-rate 2e-2
                      :learning-rate-decay-policy LearningRatePolicy/Step
                      :lr-policy-decay-rate 0.1
                      :lr-policy-steps 100000
                      :regularization true
                      :l2 5e-4
                      :momentum 0.9
                      :mini-batch false)
        hidden-layer-0 (conv-init "cnn1" channels 96 [11 11] [4 4] [3 3] 0)
        hidden-layer-1 (layers/create-layer (LocalResponseNormalization$Builder.)
                                            :name "lrn1")
        hidden-layer-2 (max-pool "maxpool1" [3 3])
        hidden-layer-3 (conv-5x5 "cnn2" 256 [1 1] [2 2] non-zero-bias)
        hidden-layer-4 (layers/create-layer (LocalResponseNormalization$Builder.)
                                            :name "lrn2")
        hidden-layer-5 (max-pool "maxPool2" [3 3])
        hidden-layer-6 (conv-3x3 "cnn3" 384 0)
        hidden-layer-7 (conv-3x3 "cnn4" 384 non-zero-bias)
        hidden-layer-8 (conv-3x3 "cnn5" 256 non-zero-bias)
        hidden-layer-9 (max-pool "maxPool3" [3 3])
        hidden-layer-10 (fully-connected "ffn1" 4096 non-zero-bias drop-out (GaussianDistribution. 0 5e-3))
        hidden-layer-11 (fully-connected "ffn2" 4096 non-zero-bias drop-out (GaussianDistribution. 0 5e-3))
        output-layer (layers/create-layer (OutputLayer$Builder. LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
                                          :name "output"
                                          :n-out num-labels
                                          :activation Activation/SOFTMAX)
        conf (netconf/multi-layer-configuration
              list-builder
              :layer [0 hidden-layer-0]
              :layer [1 hidden-layer-1]
              :layer [2 hidden-layer-2]
              :layer [3 hidden-layer-3]
              :layer [4 hidden-layer-4]
              :layer [5 hidden-layer-5]
              :layer [6 hidden-layer-6]
              :layer [7 hidden-layer-7]
              :layer [8 hidden-layer-8]
              :layer [9 hidden-layer-9]
              :layer [10 hidden-layer-10]
              :layer [11 hidden-layer-11]
              :layer [12 output-layer]
              :backprop true
              :pretrain false
              :set-input-type (InputType/convolutionalFlat height width channels))]
    (MultiLayerNetwork. conf)))

(defn custom-model
  "用户自定义模型"
  []
  nil)

(defn- *data-iter-2-record-iter*
  "数据迭代器转换为record迭代器"
  [record-reader train-data transform scaler batch-size num-labels]
  (do (.initialize record-reader train-data transform)
      (let [data-iter (RecordReaderDataSetIterator. record-reader batch-size 1 num-labels)]
        (.fit scaler data-iter)
        (.setPreProcessor data-iter scaler)
        data-iter)))

(defn- *record-reader-init-scale-fit*
  "RecordReader读取，缩放和模型训练"
  [record-reader train-data transform scaler net batch-size num-labels epochs n-cores]
  (let [data-iter (*data-iter-2-record-iter* record-reader train-data transform scaler batch-size num-labels)]
    (.fit net (MultipleEpochsIterator. epochs data-iter n-cores))))


(defn -main
  [& opts]
  (let [log (LoggerFactory/getLogger "dl4cljExample.convolution.AnimalsClassification")
        label-maker (ParentPathLabelGenerator.)
        main-path (File. (System/getProperty "user.dir") "/resources/animals/")
        file-split (FileSplit. main-path NativeImageLoader/ALLOWED_FORMATS rng)
        path-filter (BalancedPathFilter. rng label-maker num-examples num-labels batch-size)
        input-split (.sample file-split path-filter split-train-test)
        train-data (first input-split)
        test-data (last input-split)
        transforms [(FlipImageTransform. rng) (FlipImageTransform. (Random. 123)) (WarpImageTransform. rng 42)]
        scaler (ImagePreProcessingScaler. 0 1)
        listeners (l/create-listeners (ScoreIterationListener. listener-freq))
        net (k2j/doto-keyword-2-java
             (case model-type
               "LaNet" (lanet-model)
               "AlexNet" (alexnet-model)
               "custom" (custom-model)
               (throw (InvalidInputTypeException. "Incorrect model provided.")))
             :init nil
             :set-listeners listeners)
        record-reader (ImageRecordReader. height width channels label-maker)]
    (.info log "Train model ...")
    (*record-reader-init-scale-fit* record-reader train-data nil scaler net batch-size num-labels epochs n-cores)
    (doseq [transform transforms]
      (.info log (str "Training on transformation: " (class transform)))
      (*record-reader-init-scale-fit* record-reader train-data nil scaler net batch-size num-labels epochs n-cores))
    (.info log "Evaluate model ...")
    (->> (*data-iter-2-record-iter* record-reader test-data nil scaler batch-size num-labels)
         (.evaluate net)
         #(.stats % true)
         (.info log))))
