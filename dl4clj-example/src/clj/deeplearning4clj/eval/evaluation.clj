(ns deeplearning4clj.eval.evaluation
  (:require [deeplearning4clj.utils.java.keyword-2-java :as k2j])
  (:import [org.deeplearning4j.eval Evaluation]))

(defn create-evalution
  "创建Evalution"
  [class-num]
  (Evaluation. class-num))

(defn evalution-eval
  "模型结果评价"
  ([e dataset net]
   (while (.hasNext dataset)
     (let [n (.next dataset)]
       (.eval e (.getLabels n) (.getFeatureMatrix n) net))))
  #_([e true-labels guesses]
   (.eval e true-labels guesses))
  ([e true-labels input net]
   (.eval e true-labels input net)))

(def evaluation-stats (memfn stats))
