(ns deeplearning4clj.nn.conf.layers
  (:import [org.deeplearning4j.nn.conf.layers BaseOutputLayer Layer])
  (:require [clojure.string :as s]
            [deeplearning4clj.utils.java.java-class-loader :as loader]
            [deeplearning4clj.utils.java.keyword-2-java :as k2j]))

(defmacro create-layer
  "创建神经网络层"
  [layer-builder & opts]
  `(k2j/->chain-calls ~layer-builder ~@opts :build nil))

(defmacro deflayer
  "定义神经网络网络层,如果第二个参数为关键字，则layer构建的时候为无参，否则opt1为layers的参数。如果类型为vector,则opt1的内容为layer构建的时候的多个参数，不是vector类型的话为1个参数"
  [layer-type opt1 & opts]
  (let [layer-builder (loader/nested-class (resolve layer-type) 'Builder)]
    (cond (keyword? opt1) `(create-layer (new ~layer-builder) ~opt1 ~@opts)
          (vector? opt1) `(create-layer (new ~layer-builder ~@opt1) ~@opts)
          :else `(create-layer (new ~layer-builder ~opt1) ~@opts))))
