(ns deeplearning4clj.nn.conf.layers-test
  (:use clojure.test
        [deeplearning4clj.nn.conf.layers :only [deflayer]])
  (:import [org.deeplearning4j.nn.conf.layers DenseLayer]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.nd4j.linalg.activations Activation]))

(deftest layer-test
  "测试layer创建"
  (let [layer (deflayer DenseLayer
                :n-in 2
                :n-out 4
                :activation Activation/SIGMOID
                :weight-init WeightInit/DISTRIBUTION)]
    (is (= (.getNIn layer) 2) "layer输入个数测试失败")
    (is (= (.getNOut layer) 4) "layer输出个数测试失败")
    (is (= (.. layer getActivationFn toString) "sigmoid") "激活函数测试失败")
    (is (= (.getWeightInit layer) WeightInit/DISTRIBUTION) "权重初始化测试失败")))
