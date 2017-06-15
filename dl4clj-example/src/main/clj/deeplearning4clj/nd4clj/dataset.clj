(ns deeplearning4clj.nd4clj.dataset
  (:require [deeplearning4clj.nd4clj.nd4clj :as nd4clj])
  (:import [org.nd4j.linalg.dataset DataSet]))

(defn dataset
  "创建一个DataSet，input和labels为clojure序列"
  [input labels]
  (DataSet. (nd4clj/seq-2-ndarray input) (nd4clj/seq-2-ndarray labels)))
