(ns deeplearning4clj.nn.conf.multi-layer-configuration
  (:import [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork])
  (:require [deeplearning4clj.utils.java.keyword-2-java :as k2j]))

(defmacro create-list-builder
  "创建ListBuilder"
  [& opts]
  `(k2j/->chain-calls (NeuralNetConfiguration$Builder.) ~@opts :list nil))

(defmacro multi-layer-configuration
  "创建MultiLayerConfiguration"
  [list-builder & opts]
  `(k2j/->chain-calls ~list-builder ~@opts :build nil))

(defmacro def-multi-layer-network
  "创建Network"
  [conf & opts]
  `(k2j/doto-keyword-2-java (MultiLayerNetwork. ~conf) ~@opts))
