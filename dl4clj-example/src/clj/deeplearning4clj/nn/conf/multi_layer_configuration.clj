(ns deeplearning4clj.nn.conf.multi-layer-configuration
  (:import [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder NeuralNetConfiguration$ListBuilder])
  (:require [deeplearning4clj.utils.java.keyword-2-java :as k2j]))

(defmacro create-list-builder
  "创建ListBuilder"
  [& opts]
  `(k2j/->chain-calls (NeuralNetConfiguration$Builder.) ~@opts :list nil))

(defmacro multi-layer-configuration
  "创建MultiLayerConfiguration"
  [list-builder & opts]
  `(k2j/->chain-calls ~list-builder ~@opts :build nil))

(def fit (memfn fit))
