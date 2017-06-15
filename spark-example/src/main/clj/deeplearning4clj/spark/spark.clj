(ns deeplearning4clj.spark.spark
  (:require [deeplearning4clj.utils.java.java-class-loader :as loader]
            [deeplearning4clj.utils.java.keyword-2-java :as k2j]
            [sparkling.conf :as conf]
            [sparkling.core :as spark]))

(defn spark-conf
  "创建spark conf"
  [spark-master app-name]
  (-> (conf/spark-conf)
      (conf/master spark-master)
      (conf/app-name app-name)
      spark/spark-context))

(defmacro def-training-master
  "定义spark网络的配置项"
  [master-type opt1 & opts]
  (let [master-builder (loader/nested-class (resolve master-type) 'Builder)]
    `(k2j/init-object-&-run k2j/->chain-calls ~master-builder ~opt1 ~@opts :build nil)))

(defn parallelize-dataset
  "将dataset转换为RDD。转换步骤为先将dataset中的元素依次写入list,再由list转为RDD"
  [sc dataset]
  (->> (.iterator dataset)
       iterator-seq
       (spark/parallelize sc)))

(defn def-spark-net
  "定义基于spark的神经网络"
  [])
