(ns deeplearning4clj.optimize.listeners
  (:import [java.util ArrayList]))

(defn create-listeners
  "创建listeners列表"
  [& listeners]
  (let [listener-list (ArrayList.)]
    (doseq [listener listeners] (.add listener-list listener))
    listener-list))
