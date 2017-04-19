(ns deeplearning4clj.nn.conf.layers
  (:import [org.deeplearning4j.nn.conf.layers BaseOutputLayer Layer])
  (:require [clojure.string :as s]
            [deeplearning4clj.utils.java.keyword-2-java :as k2j]))

(defmacro create-layer
  "创建神经网络层"
  [layer-builder & opts]
  `(k2j/->chain-calls ~layer-builder ~@opts :build nil))
